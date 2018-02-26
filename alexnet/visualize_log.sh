# visualize_log.sh
refresh_log() {
  while true; do
    python ~/caffe/tools/extra/parse_log.py ./log/my_model_continued.log ./log/
    sleep 5 
  done
}
refresh_log & 
sleep 1
gnuplot ./gnuplot_commands
