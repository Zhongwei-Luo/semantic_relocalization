import numpy as np
import os
import shutil

if __name__ == '__main__':
    # object_list = ['02818832', '03001627', '04256520', '04379243']
    # object_list = ['03001627']
    object_list = os.listdir("/p300/Scan2CAD/data/object_model/")
    for obj in object_list:
        dataset_path = os.path.join("/p300/Scan2CAD/data/object_model/", obj)
        validation_data = os.path.join(dataset_path, 'validation_data')
        if os.path.exists(validation_data):
            shutil.rmtree(validation_data)
            os.makedirs(validation_data)
        else:
            os.makedirs(validation_data)

        train_data = os.path.join(dataset_path, 'train_data')
        if os.path.exists(train_data):
            shutil.rmtree(train_data)
            os.makedirs(train_data)
        else:
            os.makedirs(train_data)

        vis_data = os.path.join(dataset_path, 'vis_data')
        if os.path.exists(vis_data):
            shutil.rmtree(vis_data)
            os.makedirs(vis_data)
        else:
            os.makedirs(vis_data)

        files = os.listdir(dataset_path)
        label_dict = {}
        for f in files:
            if not f[-3:] == 'npy':
                continue
            name = f[13:45]
            if name not in label_dict.keys():
                label_dict[name] = 1
            else:
                label_dict[name] += 1
        # data = np.array([label_dict[i] for i in label_dict])
        # np.savetxt('./chair_distribution.txt', data)
        # np.save('./chair_distribution', label_dict)
        nums = 0
        count = {}
        for f in files:
            if not f[-3:] == 'npy':
                continue
            name = f[13:45]
            if label_dict[name] > nums:
                if name not in count.keys():
                    count[name] = 1
                else:
                    count[name] += 1
                # source_name = os.path.join(dataset_path, f)
                # if count[name] > 1:
                #     dest_name = os.path.join(train_data, f)
                #     os.system("cp " + source_name + " " + dest_name)
                #
                # else:
                #     dest_name = os.path.join(validation_data, f)
                #     os.system("cp " + source_name + " " + dest_name)
                #
                #     dest_name = os.path.join(train_data, f)
                #     os.system("cp " + source_name + " " + dest_name)
        # temp = np.array([count[i] for i in count.keys()])
        # temp = np.median(temp)
        count1 = {}
        for f in files:
            if not f[-3:] == 'npy':
                continue
            name = f[13:45]
            if name not in count1.keys():
                count1[name] = 1
            else:
                count1[name] += 1
            source_name = os.path.join(dataset_path, f)
            if count[name] >= 5:
                v = np.random.uniform()
                if v > 0.2:
                    dest_name = os.path.join(train_data, f)
                    os.system("cp " + source_name + " " + dest_name)
                else:
                    dest_name = os.path.join(validation_data, f)
                    os.system("cp " + source_name + " " + dest_name)
            if count[name] >= 0:
                if count1[name]==1:
                    dest_name = os.path.join(vis_data, f)
                    os.system("cp " + source_name + " " + dest_name)

            # name = f[13:45]
            # if label_dict[name] > nums:
            #     if name not in count1.keys():
            #         count1[name] = 1
            #     else:
            #         count1[name] += 1
            #     source_name = os.path.join(dataset_path, f)
            #
            #     if  count1[name]/count[name] <= 0.9 and count1[name]<3 :
            #         dest_name = os.path.join(train_data, f)
            #         os.system("cp " + source_name + " " + dest_name)
            #
            #     elif count1[name]/count[name] >= 0.9 and count1[name]>=3:
            #         dest_name = os.path.join(validation_data, f)
            #         os.system("cp " + source_name + " " + dest_name)

                    # dest_name = os.path.join(train_data, f)
                    # os.system("cp " + source_name + " " + dest_name)
                


    # __import__('ipdb').set_trace()



