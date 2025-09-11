def run_per_file_vis(self, file_dict, save_name = None, vis_interval=30):
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

    scene_vertices = sampled_batch['mesh_vertices'].to(self.device)
    scene_faces = sampled_batch['mesh_faces'].to(self.device)
    
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

    pc_to_world = start_T[...,:3,:3].to(self.device) @ input_pc.unsqueeze(-1).unsqueeze(2)
    pc_to_world = pc_to_world[..., 0] + start_T[...,:3,3]

    scene_vertices_to_world = start_T[...,:3,:3].to(self.device) @ scene_vertices.unsqueeze(-1).unsqueeze(2)
    scene_vertices_to_world = scene_vertices_to_world[..., 0] + start_T[...,:3,3]

    # into single seq
    batch, seq_len, J, _ = pred_pos_to_world.shape

    for idx, info in enumerate(sampled_batch['info']):
        start_frame = int(info['start_end'][0])
        predicted_position[start_frame:start_frame+seq_len] = pred_pos_to_world[idx]
        predicted_rot[start_frame:start_frame+seq_len] = pred_rotmat[idx]
        gt_position[start_frame:start_frame+seq_len] = gt_pos_to_world[idx]
        gt_rot[start_frame:start_frame+seq_len] = gt_rotmat[idx]

    # if save_name != None:
    #     save_two_pointclouds_with_colors(predicted_position.clone().detach().reshape((-1,22,3)), gt_position.clone().detach().reshape((-1,22,3)), pc_to_world.detach().reshape(-1, 3), save_name)

    pred_aa_24 = torch.Tensor(conversions.R2A(predicted_rot))
    pred_global_orient = pred_aa_24[:, 0]                                        # [B,3]
    pred_body_pose = pred_aa_24[:, 1:].reshape(total_length, 21*3)

    gt_aa_24 = torch.Tensor(conversions.R2A(gt_rot))
    gt_global_orient = gt_aa_24[:, 0]                                        # [B,3]
    gt_body_pose = gt_aa_24[:, 1:].reshape(total_length, 21*3)

    # Betas
    betas = torch.zeros(total_length, 10)
    zero_pose = torch.zeros(total_length, 10)

    # Build SMPL layer
    pred_output = self.smplx_model(
        global_orient=pred_global_orient,   # [B,3]
        body_pose=pred_body_pose,           # [B,69]
        betas=betas,                   # [B,10]
        left_hand_pose=torch.zeros((total_length, 45)), 
        right_hand_pose=torch.zeros((total_length, 45)),
        jaw_pose=torch.zeros((total_length, 3)), 
        leye_pose=torch.zeros((total_length, 3)),
        reye_pose=torch.zeros((total_length, 3)),
        expression=torch.zeros((total_length, 10)),
    )
    pred_vertices = pred_output.vertices.detach() - pred_output.joints.detach()[:, 0:1] + predicted_position[:, 0:1]
    smplx_faces = self.smplx_model.faces
    save_two_meshes_with_colors(pred_vertices[::vis_interval], smplx_faces,
                                scene_vertices_to_world[0].detach().reshape(-1, 3),
                                scene_faces[0],
                                smpl_color = [1.0, 0.75, 0.8],
                                filename=os.path.splitext(save_name)[0] + "_pred" + os.path.splitext(save_name)[1])
    gt_output = self.smplx_model(
        global_orient=gt_global_orient,   # [B,3]
        body_pose=gt_body_pose,           # [B,69]
        betas=betas,                   # [B,10]
        left_hand_pose=torch.zeros((total_length, 45)), 
        right_hand_pose=torch.zeros((total_length, 45)),
        jaw_pose=torch.zeros((total_length, 3)), 
        leye_pose=torch.zeros((total_length, 3)),
        reye_pose=torch.zeros((total_length, 3)),
        expression=torch.zeros((total_length, 10)),
    )
    gt_vertices = gt_output.vertices.detach() - gt_output.joints.detach()[:, 0:1] + gt_position[:, 0:1]
    save_two_meshes_with_colors(gt_vertices[::vis_interval], smplx_faces,
                                scene_vertices_to_world[0].detach().reshape(-1, 3),
                                scene_faces[0],
                                smpl_color = [0.53, 0.81, 0.98],
                                filename=os.path.splitext(save_name)[0] + "_gt" + os.path.splitext(save_name)[1])



    
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
