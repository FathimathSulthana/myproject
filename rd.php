<?php
set_time_limit(0);
include("header.php");

$myFile = "input_url.txt";
$fh = fopen($myFile, 'w') or die("can't open file");
$stringData = $_POST['data'];
fwrite($fh, $stringData);
fclose($fh);


$python1 = `python phish_svm.py`;
echo "<h2>SVM</h2><pre>".$python1."</pre>";


$python2 = `python phish_rf.py`;
echo "<h2>Random Forest</h2><pre>".$python2."</pre>";



$python3 = `python compare_datasets.py`;
echo "<h2>Comparison</h2><pre>".$python3."</pre>";



echo "<h4>DATA SET COMAPARISION 1</h1><hr>";

echo "<h2></h2> <img src='graph_df1.png' />";



echo "<h4>DATA SET COMAPARISION 2</h2><img src='graph_df2.png' />";



?>
