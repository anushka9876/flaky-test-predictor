"""
Handcrafted, human-readable features for flaky-test prediction.

Why this file exists:
----------------------
A deep model like CodeBERT produces a 768-number "embedding" for a piece of
code. Those numbers are great for prediction accuracy, but individual
dimensions don't mean anything a human can read ("dimension 482 was high"
means nothing to a developer).

So alongside the embedding, we extract a small set of KEYWORD/PATTERN-BASED
features that map directly onto the root-cause categories identified in the
flaky-test literature (concurrency, order-dependency, randomness, network,
time/platform — see Parry et al.'s "A Survey of Flaky Tests" and the
multivocal review by Fatima & Nagappan). These are the features we surface
in plain English in explain.py. The embedding boosts raw accuracy; the
handcrafted features are what make the system's explanations legible.

This is a heuristic, keyword-based detector — not a full static analyzer.
That's a reasonable, honestly-stated limitation for a semester project, and
a good "future work" line in your paper (e.g. AST-based detection instead
of keyword matching).
"""

import re
from dataclasses import dataclass, field


@dataclass
class FeatureSpec:
    name: str
    category: str
    pattern: str


# Each pattern maps to one of the cause categories most commonly cited in
# the flaky-test survey literature.
FEATURE_SPECS = [
    # --- Concurrency ---
    FeatureSpec("uses_thread", "Concurrency", r"\bThread\b|\bRunnable\b"),
    FeatureSpec("uses_executor", "Concurrency", r"ExecutorService|CompletableFuture|Future<"),
    FeatureSpec("uses_synchronized", "Concurrency", r"\bsynchronized\b|\block\(|ReentrantLock"),
    FeatureSpec("uses_async_await", "Concurrency", r"\basync\b|\bawait\b"),

    # --- Timing / sleeps / timeouts ---
    FeatureSpec("uses_sleep", "Timing", r"Thread\.sleep|time\.sleep\(|setTimeout\("),
    FeatureSpec("uses_timeout", "Timing", r"\btimeout\b|Timeout\(|@Timeout"),
    FeatureSpec("uses_system_time", "Timing", r"System\.currentTimeMillis|System\.nanoTime|datetime\.now\(|Date\(\)"),

    # --- Randomness ---
    FeatureSpec("uses_random", "Randomness", r"\bRandom\(|Math\.random\(|random\.\w+\(|np\.random"),
    FeatureSpec("uses_shuffle", "Randomness", r"\.shuffle\("),

    # --- Network / external I/O ---
    FeatureSpec("uses_network", "Network/External I-O", r"HttpClient|HttpURLConnection|Socket\(|requests\.\w+\(|fetch\(|axios\."),
    FeatureSpec("uses_url", "Network/External I-O", r"https?://"),
    FeatureSpec("uses_file_io", "Network/External I-O", r"\bFileReader\b|\bFileWriter\b|open\([^)]*['\"]r['\"]"),

    # --- Order dependency / shared state ---
    FeatureSpec("uses_static_field", "Order Dependency", r"\bstatic\s+\w+\s+\w+\s*="),
    FeatureSpec("uses_before_after_class", "Order Dependency", r"@BeforeClass|@AfterClass|@BeforeAll|@AfterAll"),
    FeatureSpec("uses_shared_setup", "Order Dependency", r"@Before\b|@After\b|setUp\(|tearDown\("),

    # --- Platform dependency ---
    FeatureSpec("uses_os_check", "Platform Dependency", r"os\.name|System\.getProperty\(\"os|platform\.system\("),
    FeatureSpec("uses_file_separator", "Platform Dependency", r"File\.separator|os\.sep"),

    # --- Unordered collections (a classic flakiness source) ---
    FeatureSpec("uses_hashmap_hashset", "Unordered Collection", r"\bHashMap\b|\bHashSet\b|\bdict\(|\bset\("),
]


def extract_handcrafted_features(code: str) -> dict:
    """
    Returns a dict of {feature_name: 0 or 1} for every pattern in
    FEATURE_SPECS, based on a simple regex search over the test's source
    code text.
    """
    features = {}
    for spec in FEATURE_SPECS:
        features[spec.name] = 1 if re.search(spec.pattern, code) else 0
    return features


def feature_names() -> list:
    return [spec.name for spec in FEATURE_SPECS]


def category_for_feature(name: str) -> str:
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec.category
    return "Unknown"


if __name__ == "__main__":
    sample = """
    @Test
    public void testSomething() throws InterruptedException {
        Thread.sleep(500);
        Random r = new Random();
        int x = r.nextInt();
        assertTrue(x >= 0);
    }
    """
    print(extract_handcrafted_features(sample))
