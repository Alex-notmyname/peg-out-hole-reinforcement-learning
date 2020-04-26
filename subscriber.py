#!/usr/bin/env python

import rospy
import numpy as np
import csv
import matplotlib.pyplot as plt
from geometry_msgs.msg import WrenchStamped
from DQN_brain_position import DeepQNetwork as RL_position
from DQN_brain_Angle import DeepQNetwork as RL_angle
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from robot import Robot
import copy
import math

episode_reward_path = '/home/alex/data/7/EPISODE_REWARD.csv'
episode_state_path = '/home/alex/data/7/EPISODE_STATE.csv'
episode_distance_path = '/home/alex/data/7/EPISODE_distance.csv'
graph_path_1 = '/home/alex/data/7/reward_episode.png'
graph_path_2 = '/home/alex/data/7/done_episode.png'
graph_path_3 = '/home/alex/data/7/distance_episode.png'

MAX_EPISODE = 301
MAX_STEP = 1001

# target_z_position = 80  # subject to change
episode_reward = []
episode_done = np.zeros(MAX_EPISODE)
episode_distance = []
episode_force = []

publisher = rospy.Publisher('position_trajectory_controller/command', Float64MultiArray, queue_size=1)

rb = Robot()
RL_p = RL_position()
RL_a = RL_angle()

original_position = [342.052704, -419.636, 90, 90, 0, 180]
start_position = [342.052704, -419.636, 20, 90, 0, 180]
reset_incre = 5
reset_reverse_incre = 0.2

MAX_DISTANCE = 20
MAX_FORCE = 30
MAX_TORQUE = 1.5


class DeepNetwork:

    def callback_1(self, WrenchStamped):

        """ subscribe the WrenchStamped topic and serialize force & torque msgs as numpy
            arrays, for processing in Deep Q Network
        """

        # step = WrenchStamped.header.seq

        self.x_force = WrenchStamped.wrench.force.x
        self.y_force = WrenchStamped.wrench.force.y
        self.z_force = WrenchStamped.wrench.force.z

        self.x_torque = WrenchStamped.wrench.torque.x
        self.y_torque = WrenchStamped.wrench.torque.y
        self.z_torque = WrenchStamped.wrench.torque.z

        if self.x_force > 40 or self.y_force > 40 or self.z_force > 40:

            for i in range(10):

                rospy.signal_shutdown('DANGEROUS FORCE!!!')

                print "Dangerous!!!! Go check the peg!!!!!"

    def callback_2(self, msg):
        # print("xxx")

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2]

        self.position = np.array([self.x, self.y, self.z])

        self.A = msg.position[3]
        self.B = msg.position[4]
        self.C = msg.position[5]

        self.pose = np.array([self.A, self.B, self.C])
        # print(self.pose)

    def reset(self, pub):

        self.origin = Float64MultiArray()
        self.origin_position = copy.deepcopy(original_position)
        self.origin.data = self.origin_position   # original position_1

        while pub.get_num_connections() == 0:

            print(pub.get_num_connections())
            rospy.loginfo("waiting for connection...")
            rospy.sleep(1)

        for i in range(15):

            rospy.loginfo(self.origin)
            print pub.get_num_connections()

            self.origin.data = self.origin_position

            while not self.z == self.origin_position[2]:

                pub.publish(self.origin)
                rospy.Rate(20).sleep()

            # print "step done! Input command to continue!"

            # input()

            self.origin_position[2] -= reset_incre

        print("\noriginal position reached!\n")

    def reset_reverse(self):

        self.origin_msg = Float64MultiArray()

        self.origin_ = np.append(self.position, self.pose)
        self.origin_[0] = original_position[0]
        self.origin_[1] = original_position[1]
        self.origin_[3] = original_position[3]
        self.origin_[4] = original_position[4]
        self.origin_[5] = original_position[5]

        self.origin = copy.deepcopy(self.origin_)

        while self.origin_[2] - start_position[2] > 0:

            self.origin[2] -= reset_reverse_incre
            self.origin_msg.data = self.origin
            print self.origin

            rospy.loginfo(self.origin_msg)
            print "\nReverse resetting! Hold the button!\n"

            while not np.isclose(self.x, self.origin[0], atol=1e-06):
                publisher.publish(self.origin_msg)

            while not np.isclose(self.y, self.origin[1], atol=1e-06):
                publisher.publish(self.origin_msg)

            while not np.isclose(self.z, self.origin[2], atol=1e-06):
                publisher.publish(self.origin_msg)

            if self.z <= start_position[2]:
                break

        while self.origin_[2] - start_position[2] < 0:

            self.origin[2] += reset_reverse_incre
            self.origin_msg.data = self.origin
            print self.origin

            rospy.loginfo(self.origin_msg)
            print "\nReverse resetting! Hold the button!\n"

            while not np.isclose(self.x, self.origin[0], atol=1e-06):
                publisher.publish(self.origin_msg)

            while not np.isclose(self.y, self.origin[1], atol=1e-06):
                publisher.publish(self.origin_msg)

            while not np.isclose(self.z, self.origin[2], atol=1e-06):
                publisher.publish(self.origin_msg)

            if self.z >= start_position[2]:
                break

        print "Back to original position!"

    def reset_all(self, pub):

        origin_msg = Float64MultiArray()

        origin_ = np.append(self.position, self.pose)
        origin_[0] = original_position[0]
        origin_[1] = original_position[1]
        origin_[3] = original_position[3]
        origin_[4] = original_position[4]
        origin_[5] = original_position[5]

        origin = copy.deepcopy(origin_)

        for i in range(500):

            origin[2] += reset_incre
            origin_msg.data = origin
            print origin

            rospy.loginfo(origin_msg)
            print "\nReverse resetting! Hold the button!\n"

            while not np.isclose(self.z, origin[2], atol=1e-04):
                publisher.publish(origin_msg)

            if self.z >= original_position[2]:
                break

        print("\nAll episodes done!! Go back to original position!!\n")

    def calculate(self):

        self.ave_force = math.sqrt(self.x_force ** 2 + self.y_force ** 2 + self.z_force ** 2)
        self.ave_torque = math.sqrt(self.x_torque ** 2 + self.y_torque ** 2 + self.z_torque ** 2)

        reward_on_distance = 0.5 * (self.z - start_position[2])    # length of hole, may change
        reward_on_force = 0.25 * math.pow((self.ave_force / MAX_FORCE), 2)
        reward_on_torque = 0.25 * math.pow((self.ave_torque / MAX_TORQUE), 2)

        self.reward = reward_on_distance - (reward_on_force + reward_on_torque)
        self.reward_pose = 1 - reward_on_torque * 4
        self.done = False
        self.STOP_1 = False
        self.STOP_2 = False
        self.limit_1 = False
        self.limit_2 = False

        if (self.z - start_position[2]) > MAX_DISTANCE:
            self.done = True

        elif self.ave_force > MAX_FORCE:
            self.STOP_1 = True
            self.reward = -1

        elif self.ave_torque > MAX_TORQUE:
            self.STOP_2 = True

        # if self.x - original_position[0] > 1.2 or \
                # self.y - original_position[1] > 1.2:
            # self.limit_1 = True

        if self.A - original_position[3] > 1.2 or\
                self.B - original_position[4] > 1.2 or\
                self.C - original_position[5] > 1.2:
            self.limit_2 = True

        return self.reward, self.reward_pose, self.done, self.STOP_1, self.STOP_2

    def run_this(self):

        # reset robot to the original position
        self.reset(pub=publisher)

        for episode in range(MAX_EPISODE):

            print("\nrobot reset!\n")

            # serialize force & torque data as states
            s = np.array(
                [self.x_force, self.y_force, self.z_force, self.x_torque, self.y_torque, self.z_torque, self.z])
            s_pose = np.array([self.x_torque, self.y_torque, self.z_torque])

            print self.x_force

            done = False
            episode_reward_track = []
            episode_state_track = []

            for t in range(MAX_STEP):
                print '\nt: ', t

                # choose action
                a_p = RL_p.choose_action(s)
                a_a = RL_a.choose_action(s_pose)
                print '\nchosen action: \n', a_p, # a_a

                """Publish the action to robot side"""

                s_position = rb.position_act(action_p=a_p, s=self.position)
                s_angle = rb.pose_act(action_p=a_a, s=self.pose)

                robot_action = Float64MultiArray()
                robot_action.data = np.append(s_position, self.pose)     ##### ANGLE CHANGED!!!!!

                # print np.append(s_position, s_angle)

                # v = s_position - self.position
                # c = s_angle - self.pose

                pub = rospy.Publisher('position_trajectory_controller/command', Float64MultiArray, queue_size=20)

                rate = rospy.Rate(20)  # 20hz

                while not np.isclose(self.x, s_position[0], atol=1e-06) \
                        or not np.isclose(self.y, s_position[1], atol=1e-06) \
                        or not np.isclose(self.z, s_position[2], atol=1e-06):

                        # or not np.isclose(self.A, s_angle[0], atol=1e-06) \
                        # or not np.isclose(self.B, s_angle[1], atol=1e-06) \
                        # or not np.isclose(self.C, s_angle[2], atol=1e-06):

                    connection = pub.get_num_connections()

                    if connection:

                        # rospy.loginfo(robot_action)
                        pub.publish(robot_action)

                        rate.sleep()
                    else:

                        print "No connection!!!"

                print "\naction done\n"

                # observe state_
                s_ = np.array(
                    [self.x_force, self.y_force, self.z_force, self.x_torque, self.y_torque, self.z_torque, self.z])
                s_pose_ = np.array([self.x_torque, self.y_torque, self.z_torque])

                # calculate reward and done
                r, r_pose, done, self.stop_1, self.stop_2 = self.calculate()

                print("r = ", r)
                print self.ave_force

                # if done because of large force or torque
                #   r = -1

                # store transition
                RL_p.store_transition(s, a_p, r, s_)
                # RL_a.store_transition(s_pose, a_a, r_pose, s_pose_)
                print("\ntransition stored")

                # neural network learn
                RL_p.learn()
                # RL_a.learn()
                # print("\nNetwork uploaded\n")

                # save data
                episode_reward_track.append(r)
                history = np.append(s, a_p)
                history = np.append(history, a_a)
                history = np.append(history, s_)
                history = np.append(history, r)
                episode_state_track.append(history)

                s = s_

                if self.stop_1:
                    print("DANGEROUS FORCE!!!")

                if self.stop_2:
                    print "DANGEROUS TORQUE!!!"

                if self.limit_1:
                    print "DANGEROUS POSITION!!!"

                # if self.limit_2:
                    # print "DANGEROUS ANGLE!!!"

                if done:
                    episode_done[episode] = 1

                if self.stop_1 or self.stop_2 or done or self.limit_1 or self.limit_2 or t == MAX_STEP - 1:

                    try:
                        single_episode_reward = sum(episode_reward_track) / (t + 1)      # average reward
                    except ValueError:
                        print "t = 0 ???"
                        continue

                    episode_distance.append((self.z - start_position[2]))
                    episode_force.append(self.ave_force)

                    print("\nepisode: ", episode, "reward: ", single_episode_reward, "step: \n", t)

                    with open(episode_reward_path, 'a+', ) as reward_file:
                        writer = csv.writer(reward_file, delimiter=',')
                        writer.writerow([episode + 1, single_episode_reward, t + 1, done])

                    with open(episode_state_path, 'a+', ) as state_file:
                        writer = csv.writer(state_file, delimiter=',')
                        writer.writerow([episode_state_track])

                    with open(episode_distance_path, 'a+', ) as distance_file:
                        writer = csv.writer(distance_file, delimiter=',')
                        writer.writerow([episode + 1, (self.z - start_position[2]), self.ave_force])


                    episode_reward.append(single_episode_reward)

                    break

            # if self.stop:
                # print("DANGEROUS FORCE/TORQUE!!!")

            print "\nEpisode Done! Input command to continue\n"

            # while 1:

                # try:
                    # x = int(raw_input("\nplease input 1 !!\nThis is for avoiding program shut down accidentally!\n"))

                # except ValueError:
                    # print "Please input command!"
                    # continue

                # if x == 1:

                    # print '\nRight number! Program continue\n'

                    # break

                # else:

                    # print '\nWrong number! Please try again!\n'

            self.reset_reverse()

        RL_p.save_model()

        """Plot cost graph"""
        episode_num = np.arange(MAX_EPISODE)
        plt.plot(episode_num + 1, episode_reward)
        plt.xlabel("Episodes")
        plt.ylabel("Average reward")
        plt.savefig(graph_path_1)
        plt.show()

        plt.plot(episode_num + 1, episode_done)
        plt.xlabel("Episodes")
        plt.ylabel("Done")
        plt.savefig(graph_path_2)
        plt.show()

        plt.plot(episode_num + 1, episode_distance)
        plt.xlabel("Episodes")
        plt.ylabel("Distance")
        plt.savefig(graph_path_3)
        plt.show()

        self.reset_all(pub=publisher)

    def listener(self):
        rospy.init_node('subscriber', anonymous=False, disable_signals=True)

        rospy.Subscriber('joint_states', JointState, self.callback_2)
        rospy.Subscriber('RFT_FORCE', WrenchStamped, self.callback_1)
        self.run_this()

        rospy.spin()


if __name__ == "__main__":

    DQN = DeepNetwork()

    try:
        DQN.listener()

    except rospy.ROSInterruptException:
        pass
