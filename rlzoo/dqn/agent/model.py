import torch
from torzilla import nn
from rlzoo.zoo import feature


class QFunction(nn.Module):
    def __init__(self, ob_space, ac_space, hiddens=[256], dueling=False):
        super().__init__()

        # latent
        self.latent = feature.Feature(ob_space)

        # action
        self.action_scores = nn.MLP(
            in_features = self.latent.out_features,
            out_features = hiddens + [ac_space.n],
            activations = [nn.ReLU()] * len(hiddens) + [None],
            layer_norms = [{}] * len(hiddens) + [None],
        )

        # deuling dqn
        if dueling:
            self.dueling = nn.MLP(
                in_features = self.latent.out_features,
                out_features = hiddens + [1],
                activations = [nn.ReLU()] * len(hiddens) + [None],
                layer_norms = [{}] * len(hiddens) + [None],
            )
        else:
            self.dueling = None
    
    def forward(self, input):
        latent = self.latent(input)
        action_scores = self.action_scores(latent)
        if self.dueling:
            state_score = self.dueling(latent)
            action_score_mean = torch.mean(action_scores, axis=1, keepdim=True)
            action_score_centered = action_scores - action_score_mean
            scores = state_score + action_score_centered
        else:
            scores = action_scores
        return scores