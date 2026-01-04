    Input Image (128x128x3)
   
   ↓ Conv2D (32 filters) → Feature maps (126x126x32)
   ↓ MaxPooling (2x2) → (63x63x32)

   ↓ Conv2D (64 filters) → (61x61x64)
   ↓ MaxPooling (2x2) → (30x30x64)

   ↓ Conv2D (128 filters) → (28x28x128)
   ↓ MaxPooling (2x2) → (14x14x128)

   ↓ Flatten → (25088,)
   ↓ Dense(128) → (128,)
   ↓ Dropout(0.5)
   ↓ Dense(2, softmax) → Output: [0.85, 0.15]
