# Cellulose Configuration Management

There are many ways to configure Cellulose - Python decorator, API calls,
user defined Cellulose config files etc.

Here's how we would process the flags/values in order of precedence:

1. **Cellulose APIs**

* Explicitly provided arguments to the `@Cellulose` decorator and Cellulose APIs like `benchmark(torch_model, input, ...)` and `export(torch_model, input, ...)`.

2. **Cellulose user config file (`.cellulose_config.toml`)**

* This is a configuration file that contains config values the user would like to overwrite over
the provided defaults or undefined in Cellulose APIs (step 1).
* The Cellulose user config file must be named `.cellulose_config.toml` and the `CELLULOSE_CONFIG_PATH` environment
variable be defined if not in the `$HOME` directory.

3. **Cellulose's default config file**

* We will use the Cellulose variables as defaults.
