#txt file 0, 1 -> 80, 81 mapping for YOLO

import os

class CONFIG():
    DATA_PATH = '../data/images'

config = CONFIG()


def mapping_label():
    for file in os.listdir(config.DATA_PATH):
        if file.endswith(".txt"):  # txt 파일이라면
            with open(config.DATA_PATH + file, 'r') as f:  # 파일 열기
                label = f.read().split()  # 파일 내용을 읽고, 공백 기준으로 분리
                if len(label) == 5:
                    label[0] = '80'
                elif len(label) == 10:
                    label[0] = '80'
                    label[5] = '81'
                    
            # 수정된 내용을 다시 파일에 저장
            with open(config.DATA_PATH + file, 'w') as f:
                f.write(' '.join(label[0:5]) + '\n')
                f.write(' '.join(label[5:10]))
            
    print('DONE')

    

if __name__ == '__main__':
    mapping_label(config)