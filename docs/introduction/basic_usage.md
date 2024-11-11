# Basic Usage

Our API is designed to be simple and easy to use. Here is a basic example of how to use EngiBench to create a problem, get the dataset, and evaluate a design.

```{include} ../../README.md
:start-after: <!-- start api -->
:end-before: <!-- end api -->
```

Under the hood, the design representation is converted into a format that the simulator can understand. The simulator then evaluates or optimizes the design and returns the results. Note that the underlying simulators are often written in other languages and necessitate running in containerized environments. This is completely abstracted away from the user, who only needs to provide the design and the configuration.
