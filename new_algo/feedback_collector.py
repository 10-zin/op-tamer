import random
from .dataset_loader import DataWeigher

class FeedbackCollector:
    def __init__(self, feedback_threshold=10, history_buffer_size=100):
        self.feedback_buffer = []  # Temporary storage for new feedback
        self.pos_history_pairs = []
        self.neg_history_pairs = []
        self.history_buffer_size = history_buffer_size  # Max size of the historical data buffer
        self.pos_history_buffer_size = self.history_buffer_size//2
        self.neg_history_buffer_size = self.history_buffer_size//2
        self.D_seen = {} 
        self.data_weighter = DataWeigher(success_threshold=0.5)

        self.feedback_threshold = feedback_threshold    # Update: Initialize feedback_threshold


    def collect_feedback(self, state_action_pair, feedback):
        # Collect feedback and store in the buffer
        self.feedback_buffer.append((state_action_pair, feedback))
    
    def is_enough_feedback(self):
        return len(self.feedback_buffer) >= self.feedback_threshold
    
    def reset_live_feedback_buffer(self):
        self.feedback_buffer = []  # Clear the buffer after forming pairs
    
    def update_seen_data(self,):
        # print(self.feedback_buffer)
        # print((state, action))
        for (state, action), _ in self.feedback_buffer:
            if str((state, action)) in self.D_seen:
                self.D_seen[str((state, action))] += 1
            else:
                self.D_seen[str((state, action))] = 1

    def form_contrastive_pairs(self):
        # Form contrastive pairs from the feedback buffer and previously collected data if necessary
        positive_pairs = [pair for pair in self.feedback_buffer if pair[1] >= 0]  # Assuming positive feedback is represented by values > 0
        negative_pairs = [pair for pair in self.feedback_buffer if pair[1] < 0]  # Assuming negative feedback is represented by values <= 0

        # Shuffle the lists in place
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        print("inside contrastive")
        print(positive_pairs)
        print('---\n\n')
        print(negative_pairs)

        if len(positive_pairs)>len(negative_pairs):
            negative_pairs.append((None, None))
        
        contrastive_pairs=[]
        for i in range(max(len(positive_pairs), len(negative_pairs))):
            pos_pair= (None, None) if i >= len(positive_pairs) else positive_pairs[i]
            neg_pair= (None, None) if i >= len(negative_pairs) else negative_pairs[i]
            contrastive_pairs.append((pos_pair, neg_pair))

        # Form contrastive pairs
        # contrastive_pairs = list(zip(positive_pairs, negative_pairs))

        # If necessary, supplement with historical data
        num_pos_remaining_pairs = min(len(self.feedback_buffer) - len(contrastive_pairs), len(self.pos_history_pairs))
        num_neg_remaining_pairs = min(len(self.feedback_buffer) - len(contrastive_pairs), len(self.neg_history_pairs))

        pos_history_pairs = random.sample(self.pos_history_pairs, num_pos_remaining_pairs)
        neg_history_pairs = random.sample(self.neg_history_pairs, num_neg_remaining_pairs)

        contrastive_pairs.extend(zip(pos_history_pairs, neg_history_pairs))

        # Update the historical data buffer with the new feedback
        self.pos_history_pairs.extend(positive_pairs)
        self.neg_history_pairs.extend(negative_pairs)

        self.pos_history_pairs = self.pos_history_pairs[-self.pos_history_buffer_size:]
        self.neg_history_pairs = self.neg_history_pairs[-self.neg_history_buffer_size:]

        return contrastive_pairs
    
    def form_weighted_constrastive_pairs(self):
        contrastive_pairs = self.form_contrastive_pairs()
        # print("contrastive", contrastive_pairs)
        weighted_contrastive_pairs = self.data_weighter.weight_contrastive_pairs(self.D_seen, contrastive_pairs)
        return weighted_contrastive_pairs

    def form_triplets(self):
        # Separate the feedback into positive and negative feedback tuples
        positive_feedback = [sa_pair for sa_pair in self.feedback_buffer if sa_pair[1] > 0]
        negative_feedback = [sa_pair for sa_pair in self.feedback_buffer if sa_pair[1] <= 0]

        if len(positive_feedback) < 1 or len(negative_feedback) < 1:
            return []

        triplets = []

        # Convert feedback buffer to tuples for indexing
        buffer_as_tuples = [(tuple(state), action) for (state, action), _ in self.feedback_buffer]

        for anchor_state_action, _ in self.feedback_buffer:
            anchor_as_tuple = (tuple(anchor_state_action[0]), anchor_state_action[1])
            anchor_index = buffer_as_tuples.index(anchor_as_tuple)

            # Select the closest positive and furthest negative feedback from the anchor
            positive = min(positive_feedback, key=lambda sa_pair: abs(anchor_index - buffer_as_tuples.index((tuple(sa_pair[0][0]), sa_pair[0][1]))))
            negative = max(negative_feedback, key=lambda sa_pair: abs(anchor_index - buffer_as_tuples.index((tuple(sa_pair[0][0]), sa_pair[0][1]))))

            triplets.append((anchor_as_tuple, positive[0], negative[0]))
        print(f"triplets: {triplets}")
        return triplets