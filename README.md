<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url] -->
<!-- [![Forks][forks-shield]][forks-url] -->
<!-- [![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url] -->
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/nlpie-research/efficient-ml">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">Efficiency at Scale: Investigating the Performance of Diminutive Language Models in Clinical Tasks</h3>

  <p align="center">
    Utility of different Parameter Efficient Fine-tuning (PEFT) strategies for clinical NLP tasks with models of varying scales and domain pre-training.
    <br />
    <!-- <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a> -->
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <!-- <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul> -->
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the code for the following paper: Efficiency at Scale: Investigating the Performance of Diminutive Language Models in Clinical Tasks, find the paper [here](...). 

We explored the utility of different Parameter Efficient Fine-tuning (PEFT) strategies for clinical NLP tasks. We used the models of varying scales and domain pre-training to determine the interaction of different methods and downstream task performance. We also used the following datasets: MIMIC-III, and i2b2.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ### Built With


* [![Python][Python.org]][Python-url] 

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- GETTING STARTED -->
## Getting Started

This is a fairly simple repo which utilises the HuggingFace `transformers` and `peft` libraries to fine-tune models on clinical NLP tasks. The code is designed to be run on a local machine, and the datasets are not provided.

### Prerequisites

The key libraries required are:
```
peft
transformers
torch
datasets
evaluate
```

These are quite dynamic and may change over time. Exact package versions can be found in the `requirements.txt` file.
### Datasets

All of the clinical downstream tasks were performed on the MIMIC-III and i2b2 datasets. The MIMIC-III dataset is a large, freely-available database comprising de-identified health-related data associated with over forty thousand patients who stayed in the intensive care. 

These datasets do require data agreements and will require users to request access to the data prior to use.

### Pre-processing
We follow the pre-processing steps below to prepare datasets locally for use with the scripts provided. It is a bit clunky, but you eventually want all datasets to land in the same  global dataset directory.

#### MIMIC-III
To prepare the clinical outcome tasks: mortality prediction (MIMIC MP), length of stay prediction (MIIMC LOS), we follow the steps provided by the original authors [here](https://github.com/bvanaken/clinical-outcome-prediction#create-admission-notes-for-outcome-prediction-from-mimic-iii). 

#### i2b2
Generally speaking we follow the same pre-processing steps as the original papers for the datasets. For the i2b2 tasks we follow the same steps as provided by the facebook research group [here](https://github.com/facebookresearch/bio-lm) and subsequently [here](https://github.com/nlpie-research/Lightweight-Clinical-Transformers).

##### NER tasks
Once you have the data from the original preprocessing steps, we further process the data into the HuggingFace `datasets` format. This is done using the `datasets` library and the `transformers` library.

For example, to process the i2b2 2010 NER data, we use the following script:
```python
python load_i2b2_2010_ner.py
```

At present, you will need to change the directory paths in the script to point to the correct location of the data on your local machine, as well as the save location for the new HF dataset.


### Dataset directory structure
The directory structure for the datasets should be as follows:

```
datasets
├── I2B22010NER_hf_dataset
│   ├── dataset_dict.json
│   ├── info
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── test
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── train
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── validation
│       ├── dataset_info.json
│       └── state.json
├── i2b2-2010-RE
│   ├── dataset_dict.json
│   ├── train
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── validation
│       ├── dataset_info.json
│       └── state.json
├── i2b2-2012_hf_dataset
│   ├── dataset_dict.json
│   ├── info
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── test
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── train
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── validation
│       ├── dataset_info.json
│       └── state.json
├── i2b2-2014_hf_dataset
│   ├── dataset_dict.json
│   ├── info
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── test
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── train
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── validation
│       ├── dataset_info.json
│       └── state.json
├── icd9-triage
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── MedNLI
│   ├── dataset_dict.json
│   ├── test
│   │   ├── dataset_info.json
│   │   └── state.json
│   ├── train
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── validation
│       ├── dataset_info.json
│       └── state.json
└── mimic3-clinical-outcomes
    ├── los
    │   ├── LOS_WEEKS_adm_test.csv
    │   ├── LOS_WEEKS_adm_train.csv
    │   ├── LOS_WEEKS_adm_val.csv
    │   ├── test.csv
    │   ├── train.csv
    │   └── valid.csv
    ├── mp
    │   ├── test.csv
    │   ├── train.csv
    │   └── valid.csv
```

### Training

We use a relatively simple training script to fine-tune the models on the clinical tasks. The script is designed to be run from the command line, and the user can specify the model, task, and PEFT strategy to use. To facilitate the interchanging of datasets, we use a ```yaml``` file with relevant dataset path details etc. At present, it is essential you create this `datasets.yaml` file in the root directory of the project.

An example of the yaml file is provided below:
```yaml
mimic-mp:
  training_data_dir: /mnt/sdd/efficient_ml_data/datasets/mimic3-clinical-outcomes/mp
  eval_data_dir: /mnt/sdd/efficient_ml_data/datasets/mimic3-clinical-outcomes/mp
  data_dir: ''
  training_file: train.csv
  validation_file: valid.csv
  test_file: test.csv
  task_type: SEQ_CLS
  label_name: hospital_expire_flag
  text_column: text
  remove_columns: [text]
```



The training script can be run as follows, for example to train a model on the MIMIC-III mortality prediction task with LoRA PEFT strategy:
```python
python peft_trainer.py  --model_name_or_path "$MODEL" --peft_method "LORA" --task "mimic-mp" --log_save_dir "$WHERE_YOU_WANT_LOGS" --ckpt_save_dir "$WHERE_YOU_WANT_CHECKPOINTS" --train_batch_size 32 --eval_batch_size 32 --max_epochs 5
```


<!-- USAGE EXAMPLES -->
<!-- ## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ROADMAP
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> --> -->



<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENCE](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation

```
@misc{taylor2024efficiency,
      title={Efficiency at Scale: Investigating the Performance of Diminutive Language Models in Clinical Tasks}, 
      author={Niall Taylor and Upamanyu Ghose and Omid Rohanian and Mohammadmahdi Nouriborji and Andrey Kormilitzin and David Clifton and Alejo Nevado-Holgado},
      year={2024},
      eprint={2402.10597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

<!-- CONTACT -->
<!-- ## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/nlpie-research/efficient-ml/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
[Python-url]: https://www.python.org/
