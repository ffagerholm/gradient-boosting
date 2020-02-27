# Gradient boosting
Basic implementation of gradient boosting algorithm.
Generic gradient boosting model can be found in `gradient_boosting.py`, and used with an appropriate 
loss function it can perform (binary) classification and regression. 

Some examples of loss functions can be found in `loss_functions.py`, and example of usage can be found in `test_model.py`.

## Usage
Install requirements
```
pip install -r requirements.txt
```

Example
```python
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
```

