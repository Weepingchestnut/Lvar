import torch_fidelity


def main(save_folder, img_size: int = 256):

    if img_size == 256:
        input2 = None
        fid_statistics_file = '/home/zekun/workspace/Lvar-dev/Lvar/evaluation/imagenet/adm_in256_stats.npz'
    else:
        raise NotImplementedError
    
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder,
        input2=input2,
        fid_statistics_file=fid_statistics_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']

    print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))


if __name__ == "__main__":
    # img_save_folder = '/home/zekun/workspace/Lvar-dev/Lvar/work_dir/VAR-d30-size256_cfg-1.5_seed-0'
    img_save_folder = '/home/zekun/workspace/Lvar-dev/Lvar/work_dir/seq_gen_ddp/VAR-d30-size256_cfg-1.5_seed-0'
    
    main(img_save_folder)
