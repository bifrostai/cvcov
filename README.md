# Bifrost dev setup
- Clone https://github.com/voxel51/fiftyone into the parent folder of this repo.

# Calling operators

Pattern for calling Python operators from JavaScript:

1. Import the executeOperator function:
import { executeOperator } from "@fiftyone/operators";

2. Call Python operators using the full operator name:
executeOperator("@pluginNamespace/operatorName",
parameters)

Example:

If you have a Python operator defined like this in fiftyone.yml:
```
operators:
  - my_python_operator
```

And the corresponding Python code:
```python
class MyPythonOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="my_python_operator",
            label="My Python Operator",
        )

    def execute(self, ctx):
        # Python logic here
        return {"result": "success"}

def register(plugin):
    plugin.register(MyPythonOperator)
```

You would call it from the frontend component like:
```typescript
const [result, setResult] = useState<any>(null);

useEffect(() => {
  executeOperator("@bifrostai/cvcov/my_python_operator", {
    param1: "value1",
    param2: "value2",
  }).then((res) => setResult(res));
}, []);
```
The operator name format is:
"@namespace/plugin-name/operator-name" where:
- @namespace is your plugin namespace (like @bifrostai)      
- plugin-name is your plugin name (like cvcov)
- operator-name is the name defined in the Python
operator's config

# Plugin template description (original)

This is a template for writing and building a Javascript panel plugin in FiftyOne. Once a plugin is built (as a bundled UMD file), it can be loaded into FiftyOne.

The three most important / notable files are described below.

## 1. `fiftyone.yml`

This file is a manifest for the plugin. It contains metadata about the plugin, such as its name, description, operators, and secrets.

## 2. `package.json`

As a Javascript plugin, this file is used to manage the dependencies of the plugin. Some notable entries in this file are:

a. `main`: This is the entry point of the plugin. It is the file that will be loaded when the plugin is initialized. Make sure `@fiftyone/plugins::registerComponent()` is called in this file.

b. `fiftyone::script`: This is a custom field that is used to specify the path to the bundled UMD script that will be loaded in the panel.

c. `dependencies`: If you have any FiftyOne dependencies, you'll want to use a portal to reference them. The portal should point to the path of the dependency in the local FiftyOne repository.

d. `devDependencies`: In addition to dependencies you may introduce, add a dev dependency on `@voxel51/fiftyone-js-plugin-build`. Also make sure to also declare the peer dependencies of the [fiftyone-js-plugin-build](https://github.com/voxel51/fiftyone-js-plugin-build) package.

e. `resolutions`: Also used to resolve dependency conflicts, this is usually used in this context to resolve portal references.

## 3. `vite.config.ts`

This file is used to configure the Vite build tool. It is used to bundle the plugin into a UMD script that can be loaded in the panel. The `defineConfig` function from the `fiftyone-js-plugin-build` package is used to define the Vite configuration. The `defineConfig` function takes two arguments: the directory of the plugin and an optional configuration object. The configuration object can be used to override the default Vite configuration. For example, you can use it to specify whether or not to generate sourcemaps. The `defineConfig` function returns a Vite configuration object that can be used to build the plugin. An example of how to use the `defineConfig` function is shown below:

```js
import { defineConfig } from "@voxel51/fiftyone-js-plugin-build";
import { dirname } from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const dir = __dirname;

const myAdditionalVitePlugins = [
    // add any additional Vite plugins here
];

export default defineConfig(dir, {
  buildConfigOverride: { sourcemap: true },
  plugins: myAdditionalVitePlugins,
});
```

## Important

Before running `yarn build`, make sure you have a local copy of FiftyOne and `FIFTYONE_DIR` is set to the root of the FiftyOne repository.

See also [Fiftyone JS Plugin Build Utils](https://github.com/voxel51/fiftyone-js-plugin-build).

## Troubleshooting

### `yarn` or `yarn install` fails

1. Make sure you have the right version of `yarn` in your `PATH`. You can check the version by running `yarn --version`. If you have `corepack` installed (available by default in node v16+), you'll be auto prompted to install the right version of `yarn` when you run `yarn` or `yarn install` in the plugin directory.