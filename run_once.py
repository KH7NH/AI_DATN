from predict_service import predict_both
import numpy as np

# Ngưỡng tự đặt – có thể chỉnh
GORE_THRESHOLD = 0.5
NSFW_LABELS = ["neutral", "nsfw", "sexy"]
NSFW_FLAG_THRESHOLD = 0.5   # xác suất tối thiểu để coi là nhạy cảm


while True:
    path = input("Nhập đường dẫn ảnh (hoặc 'q' để thoát): ")

    if path.lower() == 'q':
        break

    gore, nsfw = predict_both(path)

    nsfw = np.array(nsfw)
    nsfw_idx = int(np.argmax(nsfw))
    nsfw_label = NSFW_LABELS[nsfw_idx]
    nsfw_conf = float(nsfw[nsfw_idx])

    is_gore = gore >= GORE_THRESHOLD
    is_nsfw_strict = (nsfw_label != "neutral") and (nsfw_conf >= NSFW_FLAG_THRESHOLD)

    print("\n===== KẾT QUẢ THÔ =====")
    print(f"🔥 GORE score: {gore:.4f}")
    print(f"🔞 NSFW probs: {nsfw.tolist()}  →  {nsfw_label} ({nsfw_conf:.4f})")

    # ==== KẾT LUẬN DỰA TRÊN DỰ ĐOÁN ====
    print("\n===== 🧠 KẾT LUẬN =====")
    if is_gore:
        print("⚠ Ảnh được phân loại là **GORE / KINH DỊ** (vượt ngưỡng).")
    elif is_nsfw_strict:
        print(f"⚠ Ảnh được phân loại là **NHẠY CẢM ({nsfw_label})** (vượt ngưỡng).")
    else:
        print("✅ Ảnh được coi là **AN TOÀN** (không gore, không NSFW rõ rệt).")
    print("===================\n")
