echo output_voltage_* | xargs cat > allvoltages.csv
echo output_voltage_* | xargs rm
echo measured_power_* | xargs cat > allpowers.csv
echo measured_power_* | xargs rm 

