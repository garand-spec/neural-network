import argparse
import cv2


def letterbox(img, size=224, pad_value=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    out = cv2.copyMakeBorder(
        resized,
        pad_y,
        size - new_h - pad_y,
        pad_x,
        size - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=r"test.jpg")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")

    out = letterbox(img, size=args.size)

    print("original:", img.shape[1], "x", img.shape[0])
    print("letterbox:", out.shape[1], "x", out.shape[0])

    cv2.imshow("original", img)
    cv2.imshow(f"letterbox_{args.size}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()