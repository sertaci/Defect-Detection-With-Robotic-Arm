import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


SUB_TOPIC_NAME = "/rrbot/camera1/image_raw"
NODE_NAME = "show_image"

def image_callback(data):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

    # Resmi göster
    cv2.imshow("ROS Image", cv_image)
    cv2.waitKey(1) 

def main():
    rospy.init_node(NODE_NAME)
    rospy.Subscriber(SUB_TOPIC_NAME, Image, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







# import rospy
# from cv_bridge import CvBridge
# import cv2 
# from sensor_msgs.msg import Image


# class FrameHandler():
#     def __init__(self):
#         rospy.init_node("conveyor_belt_cam_node")

#         rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.main, queue_size=1)


#         self.bridge = CvBridge()

#         rospy.spin()


#     def main(self, msg):
#         frame = self.bridge.imgmsg_to_cv2(msg)
#         print(frame.shape)
#         cv2.imwrite("aaa.png", frame)


# a = FrameHandler()        