import numpy as np
from gradient_boosting import GradientBooster
from loss_functions import BinomialDevianceLoss

# xor classification problem
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([0, 1, 1, 0])

# Fit gradient boosting model with binary cross-entropy loss
model = GradientBooster(loss_function=BinomialDevianceLoss())
model.fit(X, y)

# Predict probabilities
y_prob = model.predict(X)
# Convert to class predictions
y_pred = (y_prob > 0.5).astype(int)

# Print accuracy
print((y_pred == y).mean())