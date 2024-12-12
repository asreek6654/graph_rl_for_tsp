from tqdm import tqdm

def calculate_reinforce_loss(policy_out):
    """
    This function computes the reinforce loss.
    We keep track of rewards and log_likelihoods of actions 
    and then compute the reinforce loss by averaging the negative of their
    product together over the entire batch. 
    """
    reward = policy_out["reward"]
    log_likelihood = policy_out["log_likelihood"]

    # Compute loss
    loss = -(reward * log_likelihood).mean()
    policy_out.update(
        {
            "loss": loss,
        }
    )
    return policy_out

def train_reinforce(policy, env, optimizer, train_data_size, num_epochs=100, batch_size = 512, device='cuda'):
    """
    This function executes the training loop for the REINFORCE algorithm (which is vanilla policy gradient).
    """
    policy.to(device)
    env.to(device)
    policy.train()

    num_batches = train_data_size // batch_size

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_reward = 0.0

        for batch_idx in tqdm(range(num_batches)):
            td = env.reset(batch_size = [batch_size]).to(device)

            optimizer.zero_grad()

            # Forward pass
            out = policy(td.clone(), env=env, phase='train', calc_reward=True, return_actions=True)
            rewards = out["reward"]

            # Compute Loss
            out = calculate_reinforce_loss(policy_out = out)
            loss = out["loss"]

            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reward += rewards.mean().item()

        avg_loss = epoch_loss / num_batches
        avg_reward = epoch_reward / num_batches
        print(f"Train Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")