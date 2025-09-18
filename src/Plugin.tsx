import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import { Panel } from "./Panel";
import { State } from "@fiftyone/state";

// Add the plugin to the registry
registerComponent({
  name: "Bifrost CVCov",
  label: "Bifrost CVCov",
  component: Panel,
  type: PluginComponentType.Panel,
  activator: myActivator,
  panelOptions: {
    surfaces: "grid modal",
    isNew: true,
  },
});

// A function that returns true if the plugin should be active
function myActivator({ dataset }: { dataset: State.Dataset | null }) {
  // Example of activating the plugin in a particular context
  // return dataset.name === 'quickstart'

  return true;
}
