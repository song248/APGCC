python inference.py -c ./configs/SHHA_test.yml --image ./video/airport.png --save_image ./output/output_image.png TEST.WEIGHT ./output/SHHA_best.pth TEST.THRESHOLD 0.5
python inference.py -c ./configs/SHHA_test.yml --image ./video/airport.png --save_image ./output/output_image.png

python set_pred.py -c ./configs/SHHA_test.yml --image ./video/airport.png --save_image ./output/output_image.png
