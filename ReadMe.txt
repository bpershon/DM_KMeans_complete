Brad Pershon
Data Mining HW #5

The raw data sets "wine.csv" and "TwoDimHard.csv" should be placed in the same folder as the DM_hw5.py script.

To run the program you can type python DM_hw5.py if you have your enviroment variables configured.
If not open the DM_hw5.py script up in an IDE, i use spyder from the anaconda kit, and press run.

Once the program is running, it will prompt you to enter a value for k, where k is the number of clusters.  
The final output will be placed in wine_output.csv and hard_output.csv in the same folder. The original centroid points are chosen at random, because of this it can take a few moments for the centroids to converge. 
I had this issue occur once when i ran the program and it took 62 iterations to complete, if the iterations reach above 40, feel free to stop the program by hitting ctrl + c and restarting. Generally it takes 12 to 30 iterations to
converge.

IF there are any questions please let me know at pershon.1@osu.edu