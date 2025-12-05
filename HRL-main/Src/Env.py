import torch
import numpy as np
from EmbeddingLoader import EmbeddingLoader
from Dataloader import DataReader, TrainDataset, BatchType
from tqdm import tqdm


class Environment:
    def __init__(self, max_step, epsoid_num=10, actor_num=10, device=torch.device("cpu"),
                 phase_weight=0.5, modulus_weight=0.5, reward_gamma=1e-2, topN=10):

        self.device = device
        self.epsoid_num = epsoid_num
        self.actor_num = actor_num
        self.phase_weight = phase_weight
        self.modulus_weight = modulus_weight
        self.gamma = reward_gamma
        self.max_step = max_step
        self.topN = topN

        self.embedding_loader = EmbeddingLoader()
        self.data_reader = DataReader('Data/wn18rr')

        self.triples_memory = torch.tensor(
            self.data_reader.train_data).to(self.device)
        self.triples_unknown = torch.tensor(
            self.data_reader.valid_data).to(self.device)

        self.fact_num_memory = len(self.data_reader.train_data)
        self.fact_num_unknown = len(self.data_reader.valid_data)
        print(f"Memory facts num: {self.fact_num_memory}")
        print(f"Unknown facts num: {self.fact_num_unknown}")

        self.entity_embedding = torch.tensor(
            self.embedding_loader.entity_embedding, dtype=torch.float32).to(self.device)
        self.relation_embedding = torch.tensor(
            self.embedding_loader.relation_embedding, dtype=torch.float32).to(self.device)

        # state indicators
        self.current_entity = None
        self.current_relation = None
        self.target_entity = None
        self.position = 0
        self.now_step = 0
        self.batch_done = False
        self.step_done = None

        self.pi = 3.14159262358979323846

    def reset(self):
        """
        Reset the environment

        Return:
            now_entity_embeddings, now_relation_embeddings, now_done
        """
        self.now_step = 0

        if (self.position+self.epsoid_num) < self.triples_unknown.shape[0]:
            select_triples = self.triples_unknown[self.position:self.position+self.epsoid_num]
            self.position = self.position+self.epsoid_num
        elif self.position < self.triples_unknown.shape[0]:
            select_triples = self.triples_unknown[self.position:-1]
            self.batch_done = True
            self.position = 0

        e_h = select_triples[:, 0].view(-1, 1)
        r = select_triples[:, 1].view(-1, 1)
        e_t = select_triples[:, 2].view(-1, 1)

        self.current_entity = e_h.repeat(self.actor_num, 1)
        # TBD : initialization of current_relation
        self.current_relation = r.repeat(self.actor_num, 1)
        self.target_entity = e_t.repeat(self.actor_num, 1)

        self.step_done = ((self.current_entity == self.target_entity))
        return self.entity_embedding[self.current_entity.view(-1)], self.relation_embedding[self.current_relation.view(-1)], self.step_done

    def step(self, embedding_p):
        """
        Function:
            Step forward based on given predicted embedding.
        Parameter:
            embedding_p
        Return:
            Next entity_embedding, relation_embedding, step_done, reward, score
        """
        self.now_step += 1

        if (self.now_step > 0) and (self.now_step < self.max_step):
            min_distances_with_entities, self.current_relation = self.find_closest_entity(
                embedding_p, topN=1, is_first_step=(self.now_step == 1))
        elif self.now_step == self.max_step:
            min_distances_with_entities, self.current_relation = self.find_closest_entity(
                embedding_p, topN=self.topN, is_first_step=False)
        else:
            raise ValueError

        self.current_entity = min_distances_with_entities[:, :1, 1].to(
            self.current_entity.dtype)  # float to int
        
        self.step_done = (
            (self.current_entity == self.target_entity) | self.step_done)

        reward, score = self.reward_function(min_distances_with_entities, is_last_step=(self.now_step == self.max_step))

        return self.entity_embedding[self.current_entity.view(-1)], \
            self.relation_embedding[self.current_relation.view(-1)], \
            self.step_done, reward, score

    def find_closest_entity(self, embedding_p, topN, is_first_step):
        """
        Function:
            Find the closest entity to given embedding.
        Parameter:
            embedding_p: input embedding
            topN: required num of closest entities
            is_first_step: indicator of weather be at the first step
        """
        embedding_p_copy = embedding_p.to(self.device)

        min_distances_with_entities = torch.full((embedding_p.shape[0],
                                                  topN, 2), float('inf')).to(self.device)     # size[batch, topN, 2], [:,:,0]->dist [:,:,1]->entity
        distance = torch.full(
            (embedding_p.shape[0], 1), float('inf')).to(self.device)

        link_r = self.current_relation
        min_distances_with_entities[:, :, 1] = (
            self.current_entity).repeat(1, topN)

        e_h_tensor = torch.full((embedding_p.shape[0], 1), 1).to(self.device)
        r_tensor = torch.full((embedding_p.shape[0], 1), 1).to(self.device)
        e_t_tensor = torch.full((embedding_p.shape[0], 1), 1).to(self.device)

        for triple in tqdm(self.triples_memory):
            # init
            e_h, r, e_t = triple
            if is_first_step:
                is_now_e = (self.current_entity == e_h_tensor *
                            e_h) & (self.target_entity != e_t_tensor*e_t)
            else:
                is_now_e = (self.current_entity == e_h_tensor *
                            e_h)        # perhap not rigorous?

            # calculate
            distance[is_now_e] = self.embedding_distance(embedding_p_copy,
                                                         self.entity_embedding[e_t_tensor*e_t].view(embedding_p.shape))[is_now_e]

            # update
            mask = (distance < min_distances_with_entities[:, -1:, 0])
            min_distances_with_entities[:, -1:, 0][mask] = distance[mask]
            min_distances_with_entities[:, -1:,
                                        1][mask] = (e_t_tensor[mask]*e_t).to(torch.float)
            link_r[mask] = r_tensor[mask]*r

            # sort
            idx = torch.argsort(min_distances_with_entities[:, :, 0], dim=-1)
            min_distances_with_entities = torch.gather(min_distances_with_entities, 1,
                                                       idx.unsqueeze(-1).expand(-1, -1, min_distances_with_entities.size(2)))

        return min_distances_with_entities, link_r


    def embedding_distance(self, e_1, e_2):
        phase_e_1, mod_e_1 = torch.chunk(e_1, 2, dim=-1)
        phase_e_2, mod_e_2 = torch.chunk(e_2, 2, dim=-1)
        mod_dist = torch.norm((mod_e_1-mod_e_2), dim=-1).view(-1, 1)
        phase_dist = torch.sum(
            torch.abs(torch.sin((phase_e_1-phase_e_2)/(2*self.pi))), dim=-1).view(-1, 1)
        ret_dist = (mod_dist*self.modulus_weight) + \
            (phase_dist*self.phase_weight)
        return (ret_dist)

    def reward_function(self, min_distances_with_entities, is_last_step=False):

        reward = torch.where(self.step_done, 2.0, 0.0) - self.gamma * (self.embedding_distance(
            (self.entity_embedding[self.current_entity]).squeeze(),
            (self.entity_embedding[self.target_entity]).squeeze()))     # need to be optimized

        if is_last_step:
            score = []
            hits = (min_distances_with_entities[:,:,1] == self.target_entity)
            hits[:,:1] = hits[:,:1] | self.step_done 
            hits = torch.where(hits, 1, 0)
            for i in range(min_distances_with_entities.shape[1]):
                score.append(torch.sum(torch.where((torch.sum(hits[:,:(i+1)], dim=-1) != 0), 1., 0.)).item())
            score=torch.tensor(score)

        else: 
            score = None

        return reward, score


if __name__ == '__main__':
    env = Environment(max_step=1)
    # e, r, d = env.reset()
    # print(e.shape, r.shape, d.shape)
    # h = torch.randint(low=0, high=2000, size=[10,1])
    # r = torch.randint(low=0, high=2000, size=[10,1])
    # t = torch.randint(low=0, high=2000, size=[10,1])
    p = torch.randn(100, 1000)
    env.reset()
    _, _, done, r, score = env.step(p)
    print(done.view(-1), r.view(-1), score)
