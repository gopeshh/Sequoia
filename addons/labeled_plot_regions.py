from dataclasses import dataclass
from experiment import ExperimentBase
from contextlib import contextmanager
from utils.plotting import PlotSectionLabel
from typing import List
from simple_parsing import list_field


@dataclass  # type: ignore
class LabeledPlotRegionsAddon(ExperimentBase):
    # TODO: Use a list of these objects to add annotated regions in the plot
    # enclosed by vertical lines with some text, for instance "task 0", etc.
    plot_sections: List[PlotSectionLabel] = list_field(init=False)

    @contextmanager
    def plot_region_name(self, description: str):
        start_step = self.global_step
        plot_section_label = PlotSectionLabel(
            start_step=start_step,
            stop_step=None,
            description=description,
        )
        self.plot_sections.append(plot_section_label)
        yield
        plot_section_label.stop_step = self.global_step
        
