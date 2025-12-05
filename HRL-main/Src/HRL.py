import Env
import PN
import Actor
import torch
from tqdm import tqdm



class Trainer:
    def __init__(self, epsoid_size=500, actor_num=30, max_step=3, lr=1e-3, max_epoch=1000,\
                nn_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),\
                env_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.nn_device = nn_device
        self.env_device = env_device
        self.epsoid_size = epsoid_size
        self.actor_num = actor_num
        self.max_step = max_step
        self.max_epoch = max_epoch
        self.lr = lr
        self.topN = 10
        
        self.low_policy = PN.LowPolicyNetwork().to(self.nn_device)
        self.high_policy = PN.HighPolicyNetwork().to(self.nn_device)
        self.env = Env.Environment(max_step=self.max_step ,epsoid_num=self.epsoid_size, \
            actor_num=self.actor_num, device=self.env_device, topN=self.topN)

        
        self.parameters = list(self.high_policy.parameters()) + list(self.low_policy.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)

    def train_one_epoch(self):
        epoch_returns = 0
        epoch_reward = 0
        metrics = torch.zeros(self.topN)
        self.env.batch_done = False
        while not self.env.batch_done: # check if the whole set has been enrolled

            # 1. reset enviroment
            print("Resetting epsoid......")
            returns_high = []
            returns_low  = []
            e_trace = []
            r_trace = []
            log_probs_high = []
            log_probs_low  = []
            rewards = []
            e_now, r_now, _ = self.env.reset()
            r_q = r_now
            e_trace.append(e_now)
            r_trace.append(r_now)

            # 2. implement multi-hop
            for step in range(self.max_step):
                print("Reasoning......")
                
                # 2.1 high level reasoning
                m_dist = self.high_policy(torch.stack(e_trace, dim=0).to(self.nn_device), \
                    torch.stack(r_trace, dim=0).to(self.nn_device), r_q.to(self.nn_device))
                option, log_prob_high = Actor.sample_from_multivariate_normal(m_dist)
                
                # 2.2 low level reasonging
                e_dist = self.low_policy(torch.stack(e_trace, dim=0).to(self.nn_device), \
                    torch.stack(r_trace, dim=0).to(self.nn_device), r_q.to(self.nn_device))
                action, log_prob_low = Actor.sample_from_multivariate_normal(e_dist)
                
                # 2.3 take the action
                pred = Actor.predict_from_option_and_action(option, action)
                
                # 2.4 observe reward and new state
                print(f"Running {step+1}th step......")
                e_now, r_now, _, reward, score = self.env.step(pred)
                
                # 2.5 update trace
                e_trace.append(e_now)
                r_trace.append(r_now)
                log_probs_high.append(log_prob_high)
                log_probs_low.append(log_prob_low)
                rewards.append(reward)
                if score != None:
                    metrics = metrics + score

            # 3. calculate Reward-to-Go & loss
            G = torch.zeros(rewards[0].shape).to(self.nn_device)
            for reward in reversed(rewards):
                G = G + reward.to(self.nn_device)
                returns_high.insert(0, G)
                returns_low.insert(0, G)

            returns_high = torch.cat(returns_high)
            returns_low = torch.cat(returns_low)
            loss_high = torch.sum(-returns_high * torch.cat(log_probs_high))
            loss_low = torch.sum(-returns_low * torch.cat(log_probs_low))
            epoch_returns += (loss_high+loss_low)
            (loss_high+loss_low).backward()
            epoch_reward += torch.sum(torch.cat(rewards, dim=0))


        # self.optimizer.zero_grad()
        loss = epoch_returns

        # if torch.isnan(loss):
        #     print("Loss is NaN")
        #     loss.backward(retain_graph=True)
        #     print("Computational Graph:")
        #     print(loss.grad_fn)
        # else:
        #     loss.backward()
        #     self.optimizer.step()

        metrics = (metrics / (self.actor_num*self.env.fact_num_unknown))*100.0

        return loss, epoch_reward, metrics

    def train(self):
        for epoch in range(self.max_epoch):
            loss, reward, metrics = self.train_one_epoch()
            print(f"Epoch {epoch+1} done \t loss:{loss} \t reward:{reward} \t hits:{metrics}%")




if __name__ == '__main__':
    trainer = Trainer(max_epoch=500, lr=1e-3, epsoid_size=500, max_step=5, actor_num=10, nn_device=torch.device("cpu"))
    trainer.train()


"""
TBD:    1.  reward function should show not only accomplishment but also the distance
        2.  normalized action                                                                       OK
        3.  log recording                                                                        
        4.  computing effeciency
        5.  compute max step according to KG property
        6.  agent cannot seen required triples                                                      OK
        7.  comment on classes to declear the usage and parameters (Env especially)
        8.  ram occupation of LSTM                                                                  NA
        9.  ranking score based on Hit@N
        10. running different epsoids on different GPU                                              NA
        11. test & valid mode
        12. to save the model
        13. args management
        14. file construction
"""