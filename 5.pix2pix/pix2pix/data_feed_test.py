import data_feeder
import seg2face_data_maker

celeba_image = 'D:/dataset/celebA/image'
celeba_bbox = 'D:/dataset/celebA/list_bbox_celeba.csv'
lmd_model = './shape_predictor_68_face_landmarks.dat'

def main():
    d_feeder = data_feeder.CelebABboxFeeder(celeba_image, celeba_bbox, 0.1)
    seg2face = seg2face_data_maker.Seg2FaceDataMaker(lmd_model, (128, 128), 1.2)

    img_list, bbox_list = d_feeder.get_train(1)
    seg2face.make(img_list[0], bbox_list[0])

if __name__ == '__main__':
    main()