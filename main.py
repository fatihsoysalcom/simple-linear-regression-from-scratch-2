import random
import math

# 1. Data Generation - Analogous to 'Data Collection & Preparation' in a real AI project.
# This function represents the 'input' stage, a common software engineering concept.
def generate_synthetic_data(num_samples=100, noise_level=5):
    """Generates synthetic data for linear regression: y = 2*x + 5 + noise."""
    data = []
    for _ in range(num_samples):
        x = random.uniform(0, 100)
        true_y = 2 * x + 5
        y = true_y + random.uniform(-noise_level, noise_level) # Add some noise
        data.append((x, y))
    return data

# 2. Model Definition & Training - Core 'AI Logic' encapsulated.
# This function represents the 'processing logic' and 'learning' stage.
# It demonstrates iterative refinement, a common pattern in both AI and traditional software (e.g., optimization algorithms).
def train_model(data, learning_rate=0.0001, epochs=2000):
    """Trains a simple linear regression model using gradient descent."""
    # Initialize model parameters (weights and bias) - these are the 'state' of our model.
    weight = 0.0
    bias = 0.0

    n = len(data)

    for epoch in range(epochs):
        # Gradients for weight and bias
        dw = 0.0
        db = 0.0
        for x, y in data:
            # Prediction for current parameters
            y_pred = weight * x + bias
            # Loss calculation (Mean Squared Error derivative)
            dw += -2 * x * (y - y_pred)
            db += -2 * (y - y_pred)

        # Update parameters using gradient descent
        # This is the 'learning' part, where the model adapts.
        weight -= learning_rate * (dw / n)
        bias -= learning_rate * (db / n)

        # Optional: Print loss every N epochs to monitor training progress
        if epoch % (epochs // 10) == 0:
            loss = sum((y - (weight * x + bias))**2 for x, y in data) / n
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    # Return the learned parameters - the 'trained model'.
    return weight, bias

# 3. Prediction/Inference - Using the trained model.
# This is analogous to a 'service' or 'API endpoint' in traditional software,
# where the trained logic is applied to new inputs.
def predict(x, weight, bias):
    """Makes a prediction using the trained linear regression model."""
    return weight * x + bias

# 4. Model Evaluation - Assessing performance.
# This is crucial for 'Quality Assurance' and 'Testing' in any software project.
def evaluate_model(data, weight, bias):
    """Calculates the Mean Squared Error (MSE) of the model on given data."""
    total_squared_error = 0
    for x, y_true in data:
        y_pred = predict(x, weight, bias)
        total_squared_error += (y_true - y_pred)**2
    mse = total_squared_error / len(data)
    return mse

# 5. Main execution flow - Orchestrates the AI project stages.
# This acts as the 'main application logic' or 'project management' layer,
# demonstrating the sequence of operations in a typical AI pipeline.
def main():
    print("Starting AI project simulation...")

    # Step 1: Data Acquisition and Preparation
    print("\n[Stage 1: Data Acquisition & Preparation]")
    training_data = generate_synthetic_data(num_samples=100)
    print(f"Generated {len(training_data)} synthetic data points.")

    # Step 2: Model Training
    print("\n[Stage 2: Model Training]")
    # The 'train_model' function encapsulates the complex learning algorithm.
    # This separation of concerns is a key software engineering principle.
    learned_weight, learned_bias = train_model(training_data, learning_rate=0.0001, epochs=2000)
    print(f"Training complete. Learned parameters: Weight={learned_weight:.2f}, Bias={learned_bias:.2f}")

    # Step 3: Model Evaluation
    print("\n[Stage 3: Model Evaluation]")
    # Evaluating the model's performance is like 'testing' in traditional software.
    mse = evaluate_model(training_data, learned_weight, learned_bias)
    print(f"Model Mean Squared Error on training data: {mse:.2f}")

    # Step 4: Prediction/Inference (Using the model)
    print("\n[Stage 4: Prediction/Inference]")
    test_x_values = [10, 50, 90]
    print("Making predictions for new data points:")
    for x_val in test_x_values:
        predicted_y = predict(x_val, learned_weight, learned_bias)
        print(f"  Input X: {x_val:.2f}, Predicted Y: {predicted_y:.2f}")

    print("\nAI project simulation finished.")

if __name__ == "__main__":
    main()
