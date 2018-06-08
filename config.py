landmarks = ['neckline_left','neckline_right','center_front','shoulder_left','shoulder_right','armpit_left','armpit_right','waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left','top_hem_right','waistband_left','waistband_right','hemline_left','hemline_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']

symmetry = [[0, 1], [3, 4], [5, 6], [7, 8], [9, 11], [10, 12], [13, 14], [15, 16], [17, 18], [20, 22], [21, 23]]

calculable_dict = {'blouse':[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0],
              'dress':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 0],
              'outwear':[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0],
              'skirt':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 0, 0, 0, 0, 0],
              'trousers':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       0, 0, 1, 1, 1, 1, 1]}

def check():
    
    for sp in symmetry:
        left, right = sp
        left = landmarks[left]
        right = landmarks[right]
        if not ('left' in left and 'right' in right):
            print left, right
        left = left.split('_')[0]
        right = right.split('_')[0]
        if left != right:
            print left, right
    for category, landmark in calculable_dict.items():
        print category
        for index, cable in enumerate(landmark):
            if cable:
                print '\t', landmarks[index]

if __name__ == '__main__':
    check()