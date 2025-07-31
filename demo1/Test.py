# test_camera_oc.py
import cv2
import torch
from torchvision import transforms
from Mdoel import HandAutoencoder
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HandAutoencoder().to(device)
model.load_state_dict(torch.load("model/autoencoder.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

THRESHOLD = 0.0205  # reconstruction error threshold

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(img_tensor)
        loss = F.mse_loss(recon, img_tensor).item()
        elapsed_ms = (time.time() - start_time) * 1000

    if loss < THRESHOLD:
        text = f"EL YOK. Loss: {loss:.4f} | Sure: {elapsed_ms:.1f} ms"
        color = (0, 255, 0)
    else:
        text = f"EL YOK. Loss: {loss:.4f}"
        color = (0, 0, 255)

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("RefleksNet One-Class", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
