# class DatasetManager:
#     def __init__(self):
#         self.D_feedback = []  # Stores feedback data
#         self.D_contrast = []  # Stores contrastive pairs
#         self.D_seen = {}      # Tracks frequency of state-action pairs

#     def add_feedback(self, feedback_data):
#         self.D_feedback.extend(feedback_data)

#     def add_contrastive_pairs(self, contrastive_pairs):
#         self.D_contrast.extend(contrastive_pairs)

#     def update_seen(self, state_action_pairs):
#         for state, action in state_action_pairs:
#             if (state, action) in self.D_seen:
#                 self.D_seen[(state, action)] += 1
#             else:
#                 self.D_seen[(state, action)] = 1

#     def get_data(self):
#         return self.D_feedback, self.D_contrast, self.D_seen
    
class DataWeigher:
    def __init__(self, success_threshold=0.5):
        self.success_threshold = success_threshold  # Threshold to determine success

    def _calculate_weight(self, seen_data, state_action, reward):
        frequency = seen_data.get(str(state_action), 0)
        max_frequency = max(seen_data.values(), default=1)
        difficulty_weight = 1 - (frequency / max_frequency)  # Higher weight for less seen state-actions
        
        # Adjust weight based on additional criteria if necessary
        # success_weight = 1 if reward <= self.success_threshold else 0.5  # Less weight for successful actions

        # difficulty_weight = difficulty_weight * success_weight
        print("WEIGHT STATISTICS--\n\n")
        print(frequency)
        print(max_frequency)
        print(state_action)
        print(reward)
        print(seen_data)
        print("WEIGHT STATISTICS--\n\n")
        return difficulty_weight
    
    def weight_contrastive_pairs(self, seen_data, contrastive_pairs):
        weighted_contrastive_pairs = []
        # print(contrastive_pairs)
        # for (p, n) in contrastive_pairs:
        #     print(p,n)
        #     print('---\n\n')
        for (pos_pair, neg_pair) in contrastive_pairs:
            # print(pos_pair, neg_pair)
            if not pos_pair:
                neg_state_action, neg_reward = neg_pair
                pos_reward=None
                pos_state_action=None
            elif not neg_pair:
                pos_state_action, pos_reward = pos_pair
                neg_state_action=None
                neg_reward=None
            else:
                # print("---")
                # print(pos_pair, neg_pair)
                (pos_state_action, pos_reward), (neg_state_action, neg_reward) = pos_pair, neg_pair  # Extract state-action pairs

            # Assign weights based on frequency and success rate
            if not neg_reward:
                pos_weight=1.0
                neg_weight=0.0
            elif not pos_reward:
                neg_weight=1.0
                pos_weight=0.0
            else:
                pos_weight = self._calculate_weight(seen_data, pos_state_action, pos_reward) 
                neg_weight = self._calculate_weight(seen_data, neg_state_action, neg_reward)
            # if pos_weight==neg_weight==0.0:
                # exit()
            # print(pos_weight, neg_weight)
            # exit()
            weighted_contrastive_pairs.append(((pos_pair, pos_weight), (neg_pair, neg_weight)))

        return weighted_contrastive_pairs

