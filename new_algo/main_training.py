class TrainingLoop:
    def __init__(self, policy, feedback_collector, contrastive_learner, dataset_manager, environment, num_episodes=100):
        self.policy = policy
        self.feedback_collector = feedback_collector
        self.contrastive_learner = contrastive_learner
        self.dataset_manager = dataset_manager
        self.environment = environment
        self.num_episodes = num_episodes

    def run_training_loop(self):
        for episode in range(self.num_episodes):
            state = self.environment.reset()
            done = False
            while not done:
                action = self.policy.get_action(state)
                next_state, reward, done, _ = self.environment.step(action)

                # Collect feedback
                self.feedback_collector.collect_feedback((state, action), reward)

                state = next_state

            # Check if we have enough feedback to form contrastive pairs
            if len(self.feedback_collector.feedback_buffer) >= self.feedback_collector.feedback_threshold:
                contrastive_pairs = self.feedback_collector.form_contrastive_pairs()
                self.dataset_manager.add_contrastive_pairs(contrastive_pairs)

                # Calculate weights for the feedback data
                weighted_feedback = self.data_weigher.weigh_data(self.dataset_manager.D_feedback)

                # Apply contrastive learning (if applicable) and update the policy
                contrastive_updates = self.contrastive_learner.apply_contrastive_learning(self.dataset_manager.D_contrast)
                self.policy.update_policy(weighted_feedback, contrastive_updates)

                # Update D_seen with new state-action pairs
                self.dataset_manager.update_seen([pair[0] for pair in self.dataset_manager.D_feedback])

                # Clear the feedback buffer for the next round of feedback collection
                self.feedback_collector.feedback_buffer = []

            print(f"Episode {episode + 1}/{self.num_episodes} completed.")

        print("Training loop finished.")
