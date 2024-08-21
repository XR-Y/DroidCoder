package org.callgraph;

import java.util.HashMap;
import java.util.Map;

public class LifeCycleMap {
    private final Map<String, String> lifecycleMap;

    public LifeCycleMap() {
        this.lifecycleMap = new HashMap<>();
        addBiRelation("onAttach", "onDetach");
        addBiRelation("onCreate", "onDestroy");
        addBiRelation("onCreateView", "onDestroyView");
        addBiRelation("onActivityCreated", "onDestroyView");
        addBiRelation("onRestart", "onStop");
        addBiRelation("onStart", "onStop");
        addBiRelation("onResume", "onPause");
    }

    private void addBiRelation(String key, String value) {
        this.lifecycleMap.put(key, value);
        this.lifecycleMap.put(value, key);
    }

    public Map<String, String> getLifecycleMap() {
        return this.lifecycleMap;
    }
}
