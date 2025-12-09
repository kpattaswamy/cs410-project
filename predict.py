import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model_weight_path = "./bert_stock_predictor_final/pytorch_model.bin"
if os.path.exists(model_weight_path):
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
else:
    print("model weights missing")

model.to(device)
model.eval()

def predict_stock_movement(text):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    label = "UP" if prediction == 1 else "DOWN"
    confidence = max(probs)

    return {
        'prediction': label,
        'confidence': float(confidence),
        'prob_up': float(probs[1]),
        'prob_down': float(probs[0])
    }

print("\n" + "="*60)
print("Model is live, 'quit' to exit:")
while True:
    text = input("\nEnter Reddit post: ")
    if text.lower() in ['quit', 'exit', 'q']:
        break
    result = predict_stock_movement(text)
    print(f"Prediction: **{result['prediction']}** ({result['confidence']:.1%} confidence)")
    print(f"P(UP): {result['prob_up']:.1%} | P(DOWN): {result['prob_down']:.1%}\n")
