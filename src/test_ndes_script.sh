python test_ndes_new.py test_1_old 1>/dev/null &
python test_ndes_new.py test_1_new 1>/dev/null &
wait
python test_ndes_new.py test_1
rm model_1*

python test_ndes_new.py test_2_old 2>&1 1>/dev/null &
python test_ndes_new.py test_2_new 2>&1 1>/dev/null &
wait
python test_ndes_new.py test_2
rm model_2*

python test_ndes_new.py test_5_batches_old 2>&1 1>/dev/null &
python test_ndes_new.py test_5_batches_new 2>&1 1>/dev/null &
wait
python test_ndes_new.py test_5_batches
rm model_5*