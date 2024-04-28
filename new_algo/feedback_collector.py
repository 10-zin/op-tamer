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
        for state, action, _ in self.feedback_buffer:
            if (state, action) in self.D_seen:
                self.D_seen[(state, action)] += 1
            else:
                self.D_seen[(state, action)] = 1

    def form_contrastive_pairs(self):
        # Form contrastive pairs from the feedback buffer and previously collected data if necessary
        positive_pairs = [pair for pair in self.feedback_buffer if pair[1] >= 0]  # Assuming positive feedback is represented by values > 0
        negative_pairs = [pair for pair in self.feedback_buffer if pair[1] < 0]  # Assuming negative feedback is represented by values <= 0

        # Shuffle the lists in place
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)

        # Form contrastive pairs
        contrastive_pairs = list(zip(positive_pairs, negative_pairs))

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

    # Original
    # def form_contrastive_pairs(self):
    #     # Form contrastive pairs from the feedback buffer and previously collected data if necessary
    #     positive_pairs = [pair for pair in self.feedback_buffer if pair[1] >= 0]  # Assuming positive feedback is represented by values > 0
    #     negative_pairs = [pair for pair in self.feedback_buffer if pair[1] < 0]  # Assuming negative feedback is represented by values <= 0

    #     contrastive_pairs = []
    #     # min_pairs = min(len(positive_pairs), len(negative_pairs))

    #     # Forming contrastive pairs from the current buffer
    #     # for _ in range(min_pairs):
    #     #     pos_pair = positive_pairs.pop(random.randint(0, len(positive_pairs) - 1))
    #     #     neg_pair = negative_pairs.pop(random.randint(0, len(negative_pairs) - 1))
    #     #     contrastive_pairs.append((pos_pair[0], neg_pair[0]))  # Storing only the state-action part

    #     contrastive_pairs.extend((pos_pair, neg_pair) for pos_pair, neg_pair in \
    #                              zip(random.shuffle(positive_pairs), random.shuffle(negative_pairs)))

    #     # If necessary, supplement with historical data
    #     num_pos_remaining_pairs = min(self.feedback_threshold-len(contrastive_pairs), len(self.pos_history_pairs))
    #     num_neg_remaining_pairs = min(self.feedback_threshold-len(contrastive_pairs), len(self.neg_history_pairs))

    #     pos_history_pairs = random.sample(self.pos_history_pairs, num_pos_remaining_pairs)
    #     neg_history_pairs = random.sample(self.neg_history_pairs, num_neg_remaining_pairs)
        
    #     contrastive_pairs.extend((pos_pair, neg_pair) for pos_pair, neg_pair in zip(pos_history_pairs, neg_history_pairs))

    #     # Update the historical data buffer with the new feedback
    #     self.pos_history_pairs.extend(positive_pairs)
    #     self.neg_history_pairs.extend(negative_pairs)

    #     self.pos_history_pairs[-self.pos_history_buffer_size:]
    #     self.neg_history_pairs[-self.neg_history_buffer_size:]

    #     # Return or process the formed contrastive pairs
    #     return contrastive_pairs
    
    def form_weighted_constrastive_pairs(self):
        contrastive_pairs = self.form_contrastive_pairs()
        weighted_contrastive_pairs = self.data_weighter.weight_contrastive_pairs(self.D_seen, contrastive_pairs)
        return weighted_contrastive_pairs

