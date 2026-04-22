from enum import Enum, IntEnum
from pathlib import Path

ROLE_SYSTEM = 'system'
ROLE_TOOL = 'function'
ROLE_USER = 'user'
ROLE_ASSISTANT = 'assistant'
ROLE_MODEL = 'model'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / 'benchmark_set'
RESULTS_FOLDER = PROJECT_ROOT / 'results'

MODEL_GPT_4o = 'gpt-4o-2024-08-06'
MODEL_GPT_41 = 'gpt-4.1-2025-04-14'
# MODEL_GPT_mini = 'gpt-4o-mini-2024-07-18'
MODEL_GPT_mini = 'gpt-4.1-mini-2025-04-14'
MODEL_O3 = 'o3-mini-2025-01-31'
MODEL_O4 = 'o4-mini-2025-04-16'
# MODEL_GEMINI_LEG = 'gemini-1.5-pro-002'
MODEL_GEMINI_ADV = 'gemini-2.5-pro-preview-05-06'
MODEL_GEMINI_ADV2 = 'gemini-2.5-flash-preview-05-20'
MODEL_GEMINI = 'gemini-2.5-flash'
MODEL_CLAUDE = 'claude-3-5-sonnet-20241022'
MODEL_CLAUDE_mini = 'claude-3-5-haiku-20241022'
MODEL_CLAUDE_ADV3 = 'claude-3-7-sonnet-20250219'
MODEL_CLAUDE_ADV4 = 'claude-sonnet-4-20250514'
MODEL_SHER_LOCKER = 'gpt-4.1-nano'
MODEL_SHER_LOCKER_4o = 'gpt-4o'
MODEL_SHER_LOCKER_GEMINI_FLASH = 'gemini-2.5-flash'
MODEL_SHER_LOCKER_4mini = 'gpt-4o-mini-2024-07-18'


class ScoreValues(IntEnum):
    NO_MATCH = 0
    PARTIAL_MATCH = 1
    MATCH = 2

    @classmethod
    def values(cls):
        """Return a list of all enum values"""
        return [member.value for member in cls]
    
    @classmethod
    def names(cls):
        """Return a list of all enum names"""
        return [member.name for member in cls]

class TaskLabels(str, Enum):
    """Enumeration of allowed labels."""
    HEATMAPS_CONTOUR_LINES = "Heatmaps, Contour Lines"
    TASK_SET_04 = "Task Set 04"
    SPATIAL_OPERATIONS = "Spatial operations"
    TASK_SET_03 = "Task Set 03"
    PROCESS_MERGE_VISUALIZE = "Process, Merge, Visualize"
    TASK_SET_02 = "Task Set 02"
    MERGE_VISUALIZE = "Merge, Visualize"
    TASK_SET_01 = "Task Set 01"
    VAGUE = "Vague"
    HARD = "Hard"
    CONTROL = "Control question"
    GEO_TASK_SET_01 = "Geo Task Set 01"
    TERRAIN_SURFACE_ANALYSIS = "Terrain and Surface Analysis"
    BASIC_FACTOR_EXTRACTION = "Basic Factor Extraction"
    TERRAIN_STATS_PROFILE = "Terrain Statistics and Profile Analysis"
    ELEVATION_FEATURE_VIS = "Elevation Feature Recognition and Terrain Visualization"
    GEO_TASK_SET_02 = "Geo Task Set 02"
    SPATIAL_STATS_SAMPLING = "Spatial Statistics and Sampling"
    KERNEL_DENSITY_HOTSPOT = "Kernel Density and Hotspot Analysis"
    SAMPLING_BIAS_ACCURACY = "Sampling Methods, Bias, and Accuracy Assessment"
    DISTRIBUTION_CLUSTER_ANOMALY = "Distribution Patterns, Clustering, and Anomaly Detection"
    GEO_TASK_SET_03 = "Geo Task Set 03"
    VECTOR_TOPOLOGY_GEOMETRY = "Vector Topology and Geometry"
    BUFFER_ANALYSIS = "Buffer Analysis"
    VECTOR_OVERLAY_TOPO = "Vector Overlay, Intersection, and Topological Relations"
    GEOMETRIC_CALCULATION = "Geometric Calculation"
    SPATIAL_PROXIMITY_DIRECTION = "Spatial Proximity and Direction"
    GEO_TASK_SET_04 = "Geo Task Set 04"
    MAP_ALGEBRA_AND_RASTER_ANALYSIS = "Map Algebra and Raster Analysis"
    RECLASSIFICATION_CLASSIFICATION = "Reclassification and Classification"
    ZONAL_STATS_RASTER_ATTR = "Zonal Statistics and Raster Attribute Calculation"
    MAP_ALGEBRA_LAYER_INTEGRATION = "Map Algebra Operations and Layer Integration"
    GEO_TASK_SET_05 = "Geo Task Set 05"
    INTEGRATED_MODELING_SIMULATION = "Integrated Modeling and Simulation"
    SITE_SELECTION_SUITABILITY = "Site Selection Analysis and Suitability Evaluation"
    HAZARD_RISK_SIMULATION = "Hazard Risk Assessment and Simulation"
    RESOURCE_POTENTIAL_ECO = "Resource, Potential, and Ecological Assessment"
    DYNAMIC_SIM_EMERGENCY_EFFICIENCY = "Dynamic Simulation, Emergency Response, and Efficiency Analysis"

NO_LABEL = "<NO_LABEL>"
