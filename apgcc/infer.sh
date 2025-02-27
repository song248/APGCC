python inference.py -c ./configs/SHHA_test.yml --image ./video/airport.png --save_image ./output/output_image.png TEST.WEIGHT ./output/SHHA_best.pth TEST.THRESHOLD 0.5
python inference.py -c ./configs/SHHA_test.yml --image ./video/airport.png --save_image ./output/output_image.png

python inference.py --image ./video/subway.png --save_image ./output/subway_output.png

python set_pred.py --image ./video/airport.png --save_image ./output/output_image.png
