python run_styleid.py --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he --sty /data2/ranxiangyu/kidney_patch/style --output_path /data2/ranxiangyu/styleid_out/style_out

 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats 


 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid_no_injection \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats \
    --without_attn_injection


 python run_styleid_wsi.py \
    --cnt /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he \
    --sty /data2/ranxiangyu/kidney_patch/style \
    --output_path /data2/ranxiangyu/styleid_out/style_out/styleid_no_adain \
    --precomputed /data2/ranxiangyu/styleid_out/precomputed_feats \
    --without_init_adain

python adain.py --content_dir /data2/ranxiangyu/kidney_patch/patch_png/level0/22811he --style_dir /data2/ranxiangyu/kidney_patch/style --output_dir /data2/ranxiangyu/styleid_out/style_out/adain --alpha 0。5 --gpu 1 

CUDA_VISIBLE_DEVICES=1 python cyclegan.py --n_epochs 200 --batch_size 1