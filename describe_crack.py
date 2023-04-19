# @tiger
# email: 512419406@qq.com
from cv2 import cv2
import numpy as np
import pandas as pd
import random


class CrackInformation:
    def __init__(self, img):
        """input:binary img"""
        self.img = np.array(img)
        self.img[img != 0] = 1
        self.connection_mark = dict()  # <mark_num>:<[point]>
        self.mark_line()  # mark all lins

        self.img_species = self.img.copy()  # store every point's species
        self.meta_line = dict()  # store line's information
        self.init_meta_line()
        self.mark_point()  # mark all points

    def init_meta_line(self):
        """init meta_line{}"""
        for num in self.connection_mark.keys():
            self.meta_line[num] = {
                'total': [0, 0, 0, 0],
                'endpoint': [],
                'crosspoint': [],
                'joinpoint': [],
                'alonepoint': []
            }
    # def get_length(self):

    def neighbour(self, x, y):
        """
        p9(x - 1, y - 1) p2(x - 1, y) p3(x - 1, y + 1)
        p8(x, y - 1)     p1(x, y)     p4(x, y + 1)
        p7(x + 1, y - 1) p6(x + 1, y) p5(x + 1, y + 1)
        """
        p9 = self.img[x - 1, y - 1]
        p2 = self.img[x - 1, y]
        p3 = self.img[x - 1, y + 1]
        p8 = self.img[x, y - 1]
        p1 = self.img[x, y]
        p4 = self.img[x, y + 1]
        p7 = self.img[x + 1, y - 1]
        p6 = self.img[x + 1, y]
        p5 = self.img[x + 1, y + 1]
        return [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def neighbour_with_loctaion(self, x, y, img_map):
        """same to 'neighbour', but return point and location"""
        p9 = img_map[x - 1, y - 1]
        p2 = img_map[x - 1, y]
        p3 = img_map[x - 1, y + 1]
        p8 = img_map[x, y - 1]
        p1 = img_map[x, y]
        p4 = img_map[x, y + 1]
        p7 = img_map[x + 1, y - 1]
        p6 = img_map[x + 1, y]
        p5 = img_map[x + 1, y + 1]
        return {
            # 1: [p1, x, y],
            2: [p2, x - 1, y],
            3: [p3, x - 1, y + 1],
            4: [p4, x, y + 1],
            5: [p5, x + 1, y + 1],
            6: [p6, x + 1, y],
            7: [p7, x + 1, y - 1],
            8: [p8, x, y - 1],
            9: [p9, x - 1, y - 1]
        }

    def which_point(self, x, y):
        """
        :param x:
        :param y:
        :return: 1-endpoint, 2-joinpoint, 3-crosspont, 4-alonepoint, 5-more_crosspoint, 0-backpoint
        """
        ps = self.neighbour(x, y)
        species = 0
        if ps[0] == 1:
            if np.sum(ps) == 1:
                species = 4
            elif np.sum(ps) == 2:
                species = 1
            elif np.sum(ps) == 3:
                species = 2
            elif 4 <= np.sum(ps) <= 6:
                ps_ = ps.copy()
                ps_[0] = ps_[-1]
                start = ps_[0]
                one_zero = 0
                for p in ps_[1:]:
                    if (int(start) - int(p)) == 1:
                        one_zero += 1
                    start = p
                if one_zero <= 2:
                    species = 2
                else:
                    species = 3
            else:
                species = 5
        else:
            pass
        return species

    def get_one_mark(self, location):
        """find mark in coonnection dictionary"""
        mark = 0
        for key, value in self.connection_mark.items():
            if location in value:
                mark = key
        return mark

    def move_point(self, big, small):
        """move point from big mark to small mark"""
        move_points = self.connection_mark.pop(big)
        for point in move_points:
            self.connection_mark[small].append(point)

    def distance(self, p1, p2):
        """two points' distance"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def mark_line(self):
        """base on 8-neighborhood labeling algorithm"""
        connection_num = 0
        self.img[0, :] = 0  # zero border
        self.img[:, 0] = 0
        self.img[-1, :] = 0
        self.img[:, -1] = 0
        # tiger = pd.DataFrame(self.img)
        # tiger.to_excel('tiger_76.xlsx')
        w, h = self.img.shape
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                ps = self.neighbour(i, j)
                if ps[0] == 1:
                    if np.sum([ps[8], ps[1], ps[2], ps[7]]) == 0.:  # p9, p2, p3, p8 is zero
                        connection_num += 1
                        self.connection_mark[connection_num] = [[i, j]]
                    elif ps[2] == 1 and ps[7] == 1:  # p8, p3 is one
                        max_mark = max(self.get_one_mark([i, j - 1]), self.get_one_mark([i - 1, j + 1]))
                        min_mark = min(self.get_one_mark([i, j - 1]), self.get_one_mark([i - 1, j + 1]))
                        self.connection_mark[min_mark].append([i, j])
                        if max_mark > min_mark:
                            self.move_point(max_mark, min_mark)
                    elif ps[8] == 1 and ps[2] == 1:  # p9, p3 is one
                        max_mark = max(self.get_one_mark([i - 1, j - 1]), self.get_one_mark([i - 1, j + 1]))
                        min_mark = min(self.get_one_mark([i - 1, j - 1]), self.get_one_mark([i - 1, j + 1]))
                        self.connection_mark[min_mark].append([i, j])
                        if max_mark > min_mark:
                            self.move_point(max_mark, min_mark)
                    else:
                        if self.get_one_mark([i, j - 1]):  # p8
                            self.connection_mark[self.get_one_mark([i, j - 1])].append([i, j])
                        elif self.get_one_mark([i - 1, j - 1]):  # p9
                            self.connection_mark[self.get_one_mark([i - 1, j - 1])].append([i, j])
                        elif self.get_one_mark([i - 1, j]):  # p2
                            self.connection_mark[self.get_one_mark([i - 1, j])].append([i, j])
                        elif self.get_one_mark([i - 1, j + 1]):  # p3
                            self.connection_mark[self.get_one_mark([i - 1, j + 1])].append([i, j])
                # for key, value in self.connection_mark.items():
                #     print(f"{key}:{value}")
                print(f"update:{len(self.connection_mark.keys())}")
                print('----------------------')
        print(f"mark is finished! {len(self.connection_mark.keys())}")

    def mark_point(self):
        """mark point's species"""
        img_copy = self.img.copy()  # to show the effect after marking point
        img_copy[img_copy == 1] = 255
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        for num, line in self.connection_mark.items():
            print(f"----------line{num}----------")
            n_endpoint, n_joinpoint, n_crosspoint, n_alonepoint, n_more_crosspoint = 0, 0, 0, 0, 0
            for point in line:
                species = self.which_point(point[0], point[1])
                if species == 1:
                    n_endpoint += 1
                    img_copy = cv2.circle(img_copy, center=[point[1], point[0]], radius=5, color=(0, 0, 255))
                    self.img_species[point[0], point[1]] = species
                    self.meta_line[num]['endpoint'].append(point)
                    self.meta_line[num]['total'][0] = n_endpoint
                elif species == 2:
                    n_joinpoint += 1
                    self.img_species[point[0], point[1]] = species
                    self.meta_line[num]['joinpoint'].append(point)
                    self.meta_line[num]['total'][1] = n_joinpoint
                elif species == 3:
                    n_crosspoint += 1
                    img_copy = cv2.circle(img_copy, center=[point[1], point[0]], radius=5, color=(0, 255, 0))
                    self.img_species[point[0], point[1]] = species
                    self.meta_line[num]['crosspoint'].append(point)
                    self.meta_line[num]['total'][2] = n_crosspoint
                elif species == 4:
                    n_alonepoint += 1
                    self.img_species[point[0], point[1]] = species
                    self.meta_line[num]['alonepoint'].append(point)
                    self.meta_line[num]['total'][3] = n_alonepoint
                elif species == 5:
                    n_more_crosspoint += 1
                    n_crosspoint += 1
                    img_copy = cv2.circle(img_copy, center=[point[1], point[0]], radius=5, color=(255, 0, 0))
                    self.img_species[point[0], point[1]] = 3
                    self.meta_line[num]['crosspoint'].append(point)
                    self.meta_line[num]['total'][2] = n_crosspoint
            print(f"endpoint:{n_endpoint}, "
                  f"joinpoint:{n_joinpoint}, "
                  f"crosspoint:{n_more_crosspoint}/{n_crosspoint}, "
                  f"alonepoint:{n_alonepoint}")
            print(self.meta_line[num]['total'])
        cv2.imwrite('test/mark_point.jpg', img_copy)

    def get_length(self):
        """get crack's length"""
        print("----------get length---------")
        img_mark = self.img.copy()
        for line, meta in self.meta_line.items():
            if meta['total'][0] >= 2:
                # endpoint--otherpoint
                for endpoint in meta['endpoint']:
                    # mark center as 0
                    img_mark[endpoint[0], endpoint[1]] = 0
                    # get branch
                    ps = self.neighbour_with_loctaion(endpoint[0], endpoint[1], img_mark)
                    branch = []
                    branch_length = 0
                    for p in ps.values():
                        if p[0] == 1:
                            branch = p[1:]
                            branch_length += self.distance(endpoint, p[1:])
                    if not branch:
                        continue
                    flag_change = True
                    branch_new = branch.copy()
                    while flag_change:
                        ps_branch = self.neighbour_with_loctaion(branch[0], branch[1], img_mark)

                        img_mark[branch[0], branch[1]] = 0

                        for p in ps_branch.values():
                            if p[1:] in meta['endpoint'] and p[1:] != endpoint and p[0] == 1:
                                flag_change = False
                            if p[1:] in meta['crosspoint']:
                                flag_change = False

                        if not flag_change:
                            break

                        flag_find = False
                        if ps_branch[3][0] == 1:
                            img_mark[ps_branch[3][1], ps_branch[3][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[3][1:], branch)
                                branch_new = ps_branch[3][1:]
                                flag_find = True
                        if ps_branch[5][0] == 1:
                            img_mark[ps_branch[5][1], ps_branch[5][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[5][1:], branch)
                                branch_new = ps_branch[5][1:]
                                flag_find = True
                        if ps_branch[7][0] == 1:
                            img_mark[ps_branch[7][1], ps_branch[7][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[7][1:], branch)
                                branch_new = ps_branch[7][1:]
                                flag_find = True
                        if ps_branch[9][0] == 1:
                            img_mark[ps_branch[9][1], ps_branch[9][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[9][1:], branch)
                                branch_new = ps_branch[9][1:]
                                flag_find = True
                        if ps_branch[2][0] == 1:
                            img_mark[ps_branch[2][1], ps_branch[2][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[2][1:], branch)
                                branch_new = ps_branch[2][1:]
                                flag_find = True
                        if ps_branch[4][0] == 1:
                            img_mark[ps_branch[4][1], ps_branch[4][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[4][1:], branch)
                                branch_new = ps_branch[4][1:]
                                flag_find = True
                        if ps_branch[6][0] == 1:
                            img_mark[ps_branch[6][1], ps_branch[6][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[6][1:], branch)
                                branch_new = ps_branch[6][1:]
                                flag_find = True
                        if ps_branch[8][0] == 1:
                            img_mark[ps_branch[8][1], ps_branch[8][2]] = 0
                            if not flag_find:
                                branch_length += self.distance(ps_branch[8][1:], branch)
                                branch_new = ps_branch[8][1:]
                                flag_find = True
                        if branch_new == branch:
                            flag_change = False
                        else:
                            branch = branch_new.copy()
                    print(f"line{line},endpoint{endpoint}:{branch_length}")

            if meta['total'][2] >= 2:
                # crosspoint--crosspoint
                for crosspoint in meta['crosspoint']:
                    # mark center as 0
                    img_mark[crosspoint[0], crosspoint[1]] = 0
                    # get branch
                    ps = self.neighbour_with_loctaion(crosspoint[0], crosspoint[1], img_mark)
                    branchs = []
                    branch_length = []
                    for p in ps.values():
                        if p[0] == 1:
                            branchs.append(p[1:])
                            branch_length.append(self.distance(crosspoint, p[1:]))
                    if not branchs:
                        continue
                    flag_change = True
                    branch_new = branchs.copy()
                    while flag_change:
                        for num, branch in enumerate(branchs):
                            ps_branch = self.neighbour_with_loctaion(branch[0], branch[1], img_mark)

                            img_mark[branch[0], branch[1]] = 0

                            for p in ps_branch.values():
                                if p[1:] in meta['crosspoint'] and p[1:] != crosspoint and p[0] == 1:
                                    flag_change = False

                            if not flag_change:
                                continue

                            flag_find = False
                            if ps_branch[3][0] == 1:
                                img_mark[ps_branch[3][1], ps_branch[3][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[3][1:], branch)
                                    branch_new[num] = ps_branch[3][1:]
                                    flag_find = True
                            if ps_branch[5][0] == 1:
                                img_mark[ps_branch[5][1], ps_branch[5][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[5][1:], branch)
                                    branch_new[num] = ps_branch[5][1:]
                                    flag_find = True
                            if ps_branch[7][0] == 1:
                                img_mark[ps_branch[7][1], ps_branch[7][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[7][1:], branch)
                                    branch_new[num] = ps_branch[7][1:]
                                    flag_find = True
                            if ps_branch[9][0] == 1:
                                img_mark[ps_branch[9][1], ps_branch[9][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[9][1:], branch)
                                    branch_new[num] = ps_branch[9][1:]
                                    flag_find = True
                            if ps_branch[2][0] == 1:
                                img_mark[ps_branch[2][1], ps_branch[2][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[2][1:], branch)
                                    branch_new[num] = ps_branch[2][1:]
                                    flag_find = True
                            if ps_branch[4][0] == 1:
                                img_mark[ps_branch[4][1], ps_branch[4][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[4][1:], branch)
                                    branch_new[num] = ps_branch[4][1:]
                                    flag_find = True
                            if ps_branch[6][0] == 1:
                                img_mark[ps_branch[6][1], ps_branch[6][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[6][1:], branch)
                                    branch_new[num] = ps_branch[6][1:]
                                    flag_find = True
                            if ps_branch[8][0] == 1:
                                img_mark[ps_branch[8][1], ps_branch[8][2]] = 0
                                if not flag_find:
                                    branch_length[num] += self.distance(ps_branch[8][1:], branch)
                                    branch_new[num] = ps_branch[8][1:]
                                    flag_find = True
                        if branch_new == branchs:
                            flag_change = False
                        else:
                            branchs = branch_new.copy()
                    print(f"line{line},crosspoint{crosspoint}:{branch_length}")
        img_mark[img_mark != 0] = 255
        cv2.imwrite('test/img_mark.jpg', img_mark)

    def color_line(self):
        """plot line with different color after marking line"""
        img_back = self.img.copy()
        img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        for points in self.connection_mark.values():
            # random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            while points:
                point = points.pop()
                img_back[point[0], point[1]] = color
        cv2.imwrite('test/line_color.jpg', img_back)


if __name__ == "__main__":
    img = cv2.imread('test/1.png', 0)
    # tiger = pd.DataFrame(np.array(img))
    # tiger.to_excel('tigre.xlsx')
    information = CrackInformation(img)
    # information.get_length()
    information.color_line()


