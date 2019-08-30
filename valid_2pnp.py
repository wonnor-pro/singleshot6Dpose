import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def valid(datacfg, cfgfile, weightfile, outfile, test=True):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    if test:
        valid_images = options['test']
    else:
        valid_images = options['valid']
    meshname     = options['mesh']
    backupdir    = options['backup']
    name         = options['name']
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    prefix       = 'results'
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    test_width   = 416 #originally 544
    test_height  = 416 #originally 544
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    save            = False
    testtime        = True
    use_cuda        = True
    num_classes     = 1
    testing_samples = 0.0
    eps             = 1e-5
    notpredicted    = 0 
    conf_thresh     = 0.1
    nms_thresh      = 0.4
    match_thresh    = 0.5
    if save:
        makedirs(backupdir + '/test')
        makedirs(backupdir + '/test/gt')
        makedirs(backupdir + '/test/pr')

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []
    preds_trans         = []
    preds_rot           = []
    preds_corners2D     = []
    gts_trans           = []
    gts_rot             = []
    gts_corners2D       = []

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)
    # diam          = calc_pts_diameter(np.array(mesh.vertices))
    diam          = float(options['diam'])

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(valid_images, shape=(test_width, test_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),]))
    valid_batchsize = 2

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)
    if test:
        logging("--------------test file--------------")
    else:
        logging("--------------validate file--------------")
    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0
    z = np.zeros((3, 1))
    for batch_idx, (data, target) in enumerate(test_loader):
        
        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        
        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data1 = torch.unsqueeze(data[0], 0)
        data2 = torch.unsqueeze(data[1], 0)
        # print(data1.shape)
        data1 = Variable(data1, volatile=True)
        data2 = Variable(data2, volatile=True)
        t2 = time.time()
        # print(data.shape)
        
        # print(data1.shape)
        
        # Forward pass
        output1 = model(data1).data  
        output2 = model(data2).data
        t3 = time.time()
        
        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes1 = get_region_boxes(output1, conf_thresh, num_classes)
        all_boxes2 = get_region_boxes(output2, conf_thresh, num_classes)
        t4 = time.time()

        # Iterate through all images in the batch
        for i in range(output1.size(0)):
        
            # For each image, get all the predictions
            boxes1   = all_boxes1[i]
            boxes2   = all_boxes2[i]
        
            # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
            target1 = torch.unsqueeze(target[0], 0)
            target2 = torch.unsqueeze(target[1], 0)
            truths1  = target1[i].view(-1, 21)
            truths2  = target2[i].view(-1, 21)
        
            # Get how many object are present in the scene
            num_gts1 = truths_length(truths1)
            num_gts2  = truths_length(truths2)

             # Iterate through each ground-truth object
            for k in range(num_gts1):
                box_gt1        = [truths1[k][1], truths1[k][2], truths1[k][3], truths1[k][4], truths1[k][5], truths1[k][6], 
                                truths1[k][7], truths1[k][8], truths1[k][9], truths1[k][10], truths1[k][11], truths1[k][12], 
                                truths1[k][13], truths1[k][14], truths1[k][15], truths1[k][16], truths1[k][17], truths1[k][18], 1.0, 1.0, truths1[k][0]]
                best_conf_est1 = -1

                box_gt2        = [truths2[k][1], truths2[k][2], truths2[k][3], truths2[k][4], truths2[k][5], truths2[k][6], 
                                truths2[k][7], truths2[k][8], truths2[k][9], truths2[k][10], truths2[k][11], truths2[k][12], 
                                truths2[k][13], truths2[k][14], truths2[k][15], truths2[k][16], truths2[k][17], truths2[k][18], 1.0, 1.0, truths2[k][0]]
                best_conf_est2 = -1

                # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
                for j in range(len(boxes1)):
                    if (boxes1[j][18] > best_conf_est1):
                        match1         = corner_confidence9(box_gt1[:18], torch.FloatTensor(boxes1[j][:18]))
                        box_pr1        = boxes1[j]
                        best_conf_est1 = boxes1[j][18]
                
                for j in range(len(boxes2)):
                    if (boxes2[j][18] > best_conf_est2):
                        match2         = corner_confidence9(box_gt2[:18], torch.FloatTensor(boxes2[j][:18]))
                        box_pr2        = boxes2[j]
                        best_conf_est2 = boxes2[j][18]

                # Denormalize the corner predictions 
                corners2D_gt1 = np.array(np.reshape(box_gt1[:18], [9, 2]), dtype='float32')
                corners2D_gt2 = np.array(np.reshape(box_gt2[:18], [9, 2]), dtype='float32')
                corners2D_pr1 = np.array(np.reshape(box_pr1[:18], [9, 2]), dtype='float32')
                corners2D_pr2 = np.array(np.reshape(box_pr2[:18], [9, 2]), dtype='float32')
                corners2D_gt1[:, 0] = corners2D_gt1[:, 0] * 416
                corners2D_gt1[:, 1] = corners2D_gt1[:, 1] * 416
                corners2D_gt2[:, 0] = corners2D_gt2[:, 0] * 416
                corners2D_gt2[:, 1] = corners2D_gt2[:, 1] * 416
                corners2D_pr1[:, 0] = corners2D_pr1[:, 0] * 416
                corners2D_pr1[:, 1] = corners2D_pr1[:, 1] * 416
                corners2D_pr2[:, 0] = corners2D_pr2[:, 0] * 416
                corners2D_pr2[:, 1] = corners2D_pr2[:, 1] * 416
                preds_corners2D.append(corners2D_pr1)
                preds_corners2D.append(corners2D_pr2)
                gts_corners2D.append(corners2D_gt1)
                gts_corners2D.append(corners2D_gt2)

                # Compute corner prediction error
                corner_norm1 = np.linalg.norm(corners2D_gt1 - corners2D_pr1, axis=1)
                corner_norm2 = np.linalg.norm(corners2D_gt2 - corners2D_pr2, axis=1)
                corner_dist1 = np.mean(corner_norm1)
                corner_dist2 = np.mean(corner_norm2)
                errs_corner2D.append(corner_dist1)
                errs_corner2D.append(corner_dist2)


                
                # chagne the dimension of the gts from (9,2) to (1,9,2)
                
                # corners2D_gt11 = np.array([corners2D_gt1])
                # corners2D_gt22 = np.array([corners2D_gt2])
                corners2D_pr11 = np.array([corners2D_pr1])
                corners2D_pr22 = np.array([corners2D_pr2])

                corners2D_gt = corners2D_gt1
                corners2D_pr = correct(corners2D_pr11, corners2D_pr22)

                print("original:\n", corners2D_pr1)
                print("corrected:\n", corners2D_pr)

                # Compute [R|t] by pnp
                R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(internal_calibration, dtype='float32'))
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))



                if save:
                    preds_trans.append(t_pr)
                    gts_trans.append(t_gt)
                    preds_rot.append(R_pr)
                    gts_rot.append(R_gt)

                    np.savetxt(backupdir + '/test/gt/R_' + valid_files[count][-8:-3] + 'txt', np.array(R_gt, dtype='float32'))
                    np.savetxt(backupdir + '/test/gt/t_' + valid_files[count][-8:-3] + 'txt', np.array(t_gt, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr/R_' + valid_files[count][-8:-3] + 'txt', np.array(R_pr, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr/t_' + valid_files[count][-8:-3] + 'txt', np.array(t_pr, dtype='float32'))
                    np.savetxt(backupdir + '/test/gt/corners_' + valid_files[count][-8:-3] + 'txt', np.array(corners2D_gt, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr/corners_' + valid_files[count][-8:-3] + 'txt', np.array(corners2D_pr, dtype='float32'))
                
                # Compute translation error
                trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                errs_trans.append(trans_dist)
                
                # Compute angle error
                angle_dist   = calcAngularDistance(R_gt, R_pr)
                errs_angle.append(angle_dist)
                
                # Compute pixel error
                Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                proj_2d_gt   = compute_projection(vertices, Rt_gt, internal_calibration)
                proj_2d_pred = compute_projection(vertices, Rt_pr, internal_calibration) 
                norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)

                # Compute 3D distances
                transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
                transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                vertex_dist       = np.mean(norm3d)    
                errs_3d.append(vertex_dist)  

                # Sum errors
                testing_error_trans  += trans_dist
                testing_error_angle  += angle_dist
                testing_error_pixel  += pixel_dist
                testing_samples      += 1
                count = count + 1

        t5 = time.time()

    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 5
    acc         = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc5cm5deg  = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    acc3d10     = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg  = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    corner_acc  = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    if testtime:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('            eval : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t1))
        print('-----------------------------------')

    # Print test statistics
    logging('Results of {}'.format(name))
    logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
    logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )

    if save:
        predfile = backupdir + '/predictions_linemod_' + name +  '.mat'
        scipy.io.savemat(predfile, {'R_gts': gts_rot, 't_gts':gts_trans, 'corner_gts': gts_corners2D, 'R_prs': preds_rot, 't_prs':preds_trans, 'corner_prs': preds_corners2D})

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile, True)
        valid(datacfg, cfgfile, weightfile, outfile, False)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
