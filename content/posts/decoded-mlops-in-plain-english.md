---
title: "Decoded. MLOps in Plain English #1"
subtitle: "Unlock the immense potential of ML systems!"
date: 2024-02-15T12:23:37+02:00
lastmod: 2024-02-15T12:23:37+02:00
draft: false
description: "Bored of simply building your models on a jupyter notebook? You're in the right place!"

tags: ["ML in production", "DevOps", "Machine Learning", "Deployment"]
categories: ["Decoded"]
series: ["MLOps"]

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: "img/img-posts/decoded-mlops-in-plain-english/robots-scaled.jpg"
featuredImagePreview: "img/img-posts/previews/decoded-mlops-1-preview.png"

toc:
  enable: true
math:
  enable: false
lightgallery: false
license: ""

enableWordCount: false
enableReadingTime: true
enableLastMod: false

---

<!--more-->

# MLOps... is it like Machine Learning is taking over the planet, or what?

well.. kind of 😁

Seriously, look around - pretty much any cool tech these days has some magic ML sprinkled in. And guess what? This smart stuff is totally changing the game for the better, in like, a ton of areas.

---

When we first start to learn machine learning, we all probably start at the same place, that is loading some data from **Kaggle** or any other platform, then we clean, preprocess, and prepare the data to train our ML models.

We experiment with a bunch of models, see which one does the best based on our specified metrics, and then bam! We have our perfect match!

**Now What?**

If your goal was to get a new top-notch model architecture on your dataset, then you're done! However, getting your model out in the world in the hands of real users is a complete different story.

**It requires a complete shift of your mindset**.

from the `Researcher` mindset: **experimentaing with different models and ideas to develop new architectures** in order to get the best performance on a benchmark or dataset.

to the `Engineer` mindset: **developing a fully scalable, reliable system** that uses a model inside ot it to solve a problem.

## MLOps

or **Machine Learning Operations** 

is the industry's go-to toolkit for building, deploying, and managing machine learning pipelines efficiently. These best practices ensure your pipelines are ready for real-world use and easy to maintain.

the MLOps Lifecycle consists of 4 different stages:

- **SCOPING**

- **DATA**

- **MODELING**

- **MONITORING**

> Throughout the **`Decoded`: MLOps in Plain English** series we'll go through each stage one by one. 
> 
> for each stage - except scoping - I'll be posting 2-3 posts:
> 
> - **theoretical**
>   
>   Where I try to explain the different concepts and different practices of each stage
> 
> - **practical**
>   
>   where we implement the different things we learned with **TFX** (Tensorflow Extended) the tool powering almost all google's AI products

**For this tutorial we'll focus on scoping and try to highlight all of its Keypoints**. The first stage in the MLOps life cycle and the most important!

----

## SCOPING

It's the first step in any Machine learning project that involves defining your business need - the problem - and the potential ML systems - the solutions - that would fulfill that need. In simple terms, it's

- **Picking the most sensible and correct path to put effort into**.

- **Planning your way ahead**.

.

**For example,** let's say that an online shopping platform is looking to enable customers to navigate their website more easily, so it's much more easier to buy and look through the different available items.

**You could brainstorm the following different solutions**

- *create a chatbot assistant*, 
  
  that can navigate and look for the different items for them

- *create a system that analyzes how the customers navigate the website*,
  
  to predict and recommend the different places a customer might want to navigate.

- *create a recommendation system to recommend products that're closer to a customer spendings*
  
  because you'll make it easier for customers to quickly see what they can affort.

.

**After you've brainstormed the different possible solutions**, as a **MLOps Expert** you need to ask yourself the following questions

1. *What project should we work on?*

2. *What are the metrics for success?*

3. *What are the available resources?* (e.g. data, time, compute, people, etc..)

**To easily answer all of this you should use the `scoping process`**

### The Scoping Process

Here's a simple illustration

![](img/img-posts/decoded-mlops-in-plain-english/scoping-process-2.png "The Scoping Process")

as you can see, there're primarliy four steps in this process.

**If your client already has identifed a problem**, then you only need to go through the remaining 3 steps!

*Let's go through the steps!*

- Brainstorm **Business** *Problems*.
  
  In most cases, your client will already have a problem that's why they reached out to you in the first place!
  
  However, sometimes the client problem might be vague and not well-defined (e.g. Increasing Revenue) in this case you could ask your client the following question:
  
  **What are the top things that you wish working better and you think they have a direct relation to your end goal?**
  
  > In some cases, asking questions to narrow down and to define some real problems is esstantial to to all of the MLOps life cycle.

- Brainstorm **AI** *solutions*.
  
  After you have defined and narrowed down on real problems, then you should brainstorm all the solutions you can think of to solve the problems, go back to the previous example of the online shopping platform to understand more.

- Assess **Feasibility** and **Value**
  
  Once you've a set of possible solutions you need to go through them one by one and assess their feasibility and value.
  
  `Feasibilitiy`
  
  Is trying to assess if a solution can be implemented in reality, 
  
  there are a lot of dimensions that you would need to consider!
  
     - Is the required data for this solution is available?
  
     - Is there any known model architecture that was able to perform well on it?
  
     - Is it costly?
  
  **You could use differnet externel benchmarks to understand the different possibilities**
  
  You can also this table to guide you. 
  
     - cols: the required data type for a solution.
  
     - rows: whether your client already has a system or not.
  
  |          | Unstructured                 | Structured                                       |
  |:--------:| ---------------------------- | ------------------------------------------------ |
  | New      | HLP                          | Predictive features available?                   |
  | Existing | HLP + History of the project | New predictive features + History of the project |
  
  **`HLP`**: given the same data, **can a human**, perform the task?
  
  **`History of the Project`**: can give great insights whether a solution is able to achieve something or not.
  
  .
  
  **For example,** suppose your client has a recommendation system built with algorithm `A` you figured that for some problems that increasing the performance of this model to **85%** accuracy would solve the problem. 
  
  ![](img/img-posts/decoded-mlops-in-plain-english/previous-project-records.png "Previous Project Performance")
  
  You could then try to guess what the next performance will be and based on that you could reject this algorithm and try to look for another.
  
  **In this plot**, the algorithm was getting more and more flat on the 70% mark. Trying to push it to 85% seems not possible, and therefore this algorithm is rejected.
  
  **`Value`**
  
  which is simply how much can this system could contribute to solving the problem.
  
  Some solutions offer more intuitive and easy-to-implement solutions others are much harder to implement. You should need to ultimately think about the value! And not just pick a solution because it's easier!

- Set **Milestones** and **Budget**
  
  milestones are like breaking down your solutions into smaller parts, kind of creating a plan or a to-do list.
  
  **So you know** that you're on a correct and a logical track of implementing your solution.
  
  Budgeting is also one of the most important things to consider, you need to carefully allocate your financial resources wisely to ensure that your project can be successful.
  
  **Note: This final step heavily relies on the solution, model architecture and data you'll be using, so it really differs from one case to another**

> **And that's it!**
> 
> In this article, we discussed the meaning of **MLOps** and how developing models in the production environments is inherently different than just trying to achieve higher performances on specifc datasets and benchmarks. We also talked about the first and the most important step in a MLOps life cycle which is `scoping` and the `scoping process` and its different steps.

---

## Let's go!

You finished the first post of the `Decoded. MLOps in Plain English`! I hope you enjoyed reading it, and if you did, please consider sharing it with your friends and colleagues who might be interested in reading more about MLOps.

![](img/img-posts/previews/decoded-mlops-1-preview.png)

**Thanks again!**

Astrodev.