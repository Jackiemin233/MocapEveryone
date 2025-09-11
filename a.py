def run_per_file(self, file_dict, save_name = None):
    sampled_batch = file_dict
    total_length = sampled_batch['total_length']
    # create placeholder for pred pos, pred rot, gt pos and gt rot
    predicted_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
    predicted_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))
    gt_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
    gt_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))

    input_seq = sampled_batch['input_seq'].to(self.device)
    input_img = None
    input_pc = sampled_batch['scene_points'].to(self.device)

    # norm_input
    input_seq = (input_seq - self.mean) / self.std 

    output_tuple = self.model(input_seq.float(), input_img = input_img, input_pc = input_pc)

    results = self.get_loss_eval(output_tuple=output_tuple, gt_tuple=sampled_batch, \
                                get_results=False, \
                                get_loss=True, \
                                is_eval=True) 
    
    start_T = sampled_batch['head_start'].to(self.device) # Start pos

    # get pred into world coord
    pred_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['pred_pos'].unsqueeze(-1)
    pred_pos_to_world = pred_pos_to_world[...,0] + start_T[...,:3,3]
    pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
    pred_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ pred_rotmat[...,0:1,:,:]

    # get gt into world coord
    gt_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['gt_pos'].unsqueeze(-1)
    gt_pos_to_world = gt_pos_to_world[...,0] + start_T[...,:3,3]
    gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
    gt_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ gt_rotmat[...,0:1,:,:]
    
    if save_name != None:
        save_two_pointclouds_with_colors(predicted_position.clone().detach().reshape((-1,22,3)), 
                                            gt_position.clone().detach().reshape((-1,22,3)), 
                                            save_name,
                                        )

    # into single seq
    batch, seq_len, J, _ = pred_pos_to_world.shape

    for idx, info in enumerate(sampled_batch['info']):
        start_frame = int(info['start_end'][0])
        predicted_position[start_frame:start_frame+seq_len] = pred_pos_to_world[idx]
        predicted_rot[start_frame:start_frame+seq_len] = pred_rotmat[idx]
        gt_position[start_frame:start_frame+seq_len] = gt_pos_to_world[idx]
        gt_rot[start_frame:start_frame+seq_len] = gt_rotmat[idx]


    predicted_angle_np = conversions.R2A(predicted_rot.cpu().numpy())
    predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
    predicted_root_angle = predicted_angle[...,0,:] 

    gt_angle_np = conversions.R2A(gt_rot.cpu().numpy()) 
    gt_angle = torch.from_numpy(gt_angle_np).cuda().float() 
    gt_root_angle = gt_angle[...,0,:] 
            
    # after running iterations get numbers
    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
    hand_index = [20, 21]
    foot_index = [7, 8]
    eval_log = {}
    for metric in self.eval_metric:
        eval_metric = get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                hand_index,
                foot_index,
                fps=motion_constants.FPS,
                root_rel=True
            ).cpu().numpy()
        eval_log[metric] = eval_metric 
    
    # add filename
    parts = sampled_batch['filename'].split('/')
    filename = '/'.join(parts[-1:])
    eval_log['filename'] = filename
    torch.cuda.empty_cache()

    return eval_log
