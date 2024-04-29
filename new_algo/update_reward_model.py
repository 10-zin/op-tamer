import numpy as np
from utils import get_feat_frozenlake

def update_reward_model_weighted_contrastive(weighted_contrastive_pairs, theta, learning_rate, margin):
    for ((pos_pair, pos_weight), (neg_pair, neg_weight)) in weighted_contrastive_pairs:
        (pos_state, pos_action), pos_reward_feedback = pos_pair if pos_pair else (None, None), 0.0
        (neg_state, neg_action), neg_reward_feedback = neg_pair if pos_pair else (None, None), 0.0
        
        # Get feature representations
        pos_feat = get_feat_frozenlake(pos_state, pos_action) if pos_weight !=0.0 else np.zeros(theta.T.shape)
        neg_feat = get_feat_frozenlake(neg_state, neg_action) if neg_weight !=0.0 else np.zeros(theta.T.shape)
        
        # Calculate predicted rewards
        pos_pred_reward = np.dot(pos_feat, theta)
        neg_pred_reward = np.dot(neg_feat, theta)
        
        # Hybrid Loss Computation (including contrastive and regression components)
        contrastive_error = max(0, margin + (pos_pred_reward - neg_pred_reward))
        pos_regression_error = ((pos_pred_reward - pos_reward_feedback)/2) ** 2
        neg_regression_error = ((neg_pred_reward - neg_reward_feedback)/2) ** 2
        total_loss = contrastive_error + pos_weight * pos_regression_error + neg_weight * neg_regression_error
        print(contrastive_error, margin, pos_pred_reward, pos_reward_feedback, neg_pred_reward, neg_reward_feedback, pos_weight, neg_weight)
        print(total_loss)
        print('total loss\n\n')
        # exit()
        
        # Gradient for regression components (adjusted with weights)
        print("feature matrix")
        print(pos_feat)
        print(neg_feat)
        grad_regression_pos = pos_weight * (pos_pred_reward - pos_reward_feedback) * pos_feat
        grad_regression_neg = neg_weight * (neg_pred_reward - neg_reward_feedback) * neg_feat
        grad_contrastive=pos_feat-neg_feat if contrastive_error>0 else np.zeros(pos_feat)
        
        # Total Gradient is the sum of both regression gradients
        total_gradient = grad_contrastive+grad_regression_pos + grad_regression_neg
        print(total_gradient)
        
        # Update theta using the total gradient
        theta -= learning_rate * total_gradient.reshape(-1, 1)
        print(theta)

    return theta

