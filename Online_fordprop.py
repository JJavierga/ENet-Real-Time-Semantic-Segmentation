import cv2
import torch
import torch.nn as nn

from utils import decode_segmap
from models.ENet import ENet


if __name__ == '__main__':


    enet=ENet(12)

    device=torch.device('cuda:0' if torch.cuda.is_available() \
                               else 'cpu')

    checkpoint = torch.load('./ckpt-enet-100-35.71956396102905.pth')
    enet.load_state_dict(checkpoint['state_dict'])

    enet=enet.to(device)

    cam1 = cv2.VideoCapture(0)

    #cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam1.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        tmg_ = cv2.resize(frame, (512, 512), cv2.INTER_NEAREST)
        tmg = torch.tensor(tmg_).unsqueeze(0).float()
        tmg = tmg.transpose(2, 3).transpose(1, 2).to(device)

        with torch.no_grad():
            out1 = enet(tmg.float()).squeeze(0)

        out2 = out1.cpu().detach().numpy()
        segmentated=decode_segmap(out2)

        final=cv2.vconcat([frame,segmentated])
        imshow("Comparison",final)
                
        k = cv2.waitKey(10)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()