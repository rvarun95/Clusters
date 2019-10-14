# Exoplanet Prediction

Identifying and Finding Exoplanets in deep space

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install below packages.

```bash
pip install imblearn deap update_checker tqdm stopit dask[delayed] dask-ml 
pip install scikit-mdr skrebate tpot Flask flask-restful
```

## Usage

```bash
sudo docker-compose build

nohup docker-compose up &

python3 test.py

```

## Project Structure

![ref]

[ref]: https://github.com/rvarun95/Clusters/blob/master/project_structure.png


## Model evaluation 
 - When compared to Machine Learning model vs Deep learning CNN 1D model, CNN 1D model performed well based on accuracy score and in specific CNN 1D with ADAM optimizer gives prominent solution for this Exoplanet datasets
 - So Final model, Mostly works well on Deep Learning CNN 1D with Adam optimizer, In Machine learning models also works fine such as SGD classifier, Gaussian Naive Bayes Macine Learning models provide high accuracy values
 
<table border='1'>
<tr>
<th>Model Name</th>
<th>Accuray Score (in %)</th>
</tr>

<tr>
<td>Linear Model with SGD Classifier</td>
<td>99.47</td>
</tr>

<tr>
<td>Random Forest</td>
<td>99.15</td>
</tr>

<tr>
<td>Decision Tree</td>
<td>99.15</td>
</tr>

<tr>
<td>Naive Bayes</td>
<td>99.52</td>
</tr>

<tr>
<td>SG Boosting</td>
<td>99.12</td>
</tr>

<tr>
<td>ADA Boosting</td>
<td>99.12</td>
</tr>

<tr>
<td>CNN with Adam Optimizer</td>
<td>98.24</td>
</tr>

<tr>
<td>CNN with SGD Optimizer</td>
<td>99.82</td>
</tr>

<tr>
<td>CNN with SGD Optimizer, Learning Rate and Momentum</td>
<td>99.84</td>
</tr>

</table>

	
## Contributors

 
[Varun Rajendran ](https://github.com/rvarun95 "Varun's github") 

[Mahendran Mohan ](https://github.com/mahendranmohan "Mahendran's github")

[Kumaran K ](https://github.com/rvarun95 "Kumaran's github") 

[Jeya Kumar ](https://github.com/mahendranmohan "Jeyakumar's github")

[Rajan S ](https://github.com/rvarun95 "Rajan's github") 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
