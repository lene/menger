package menger.engines.scene

// optix-jni 0.5.0 types instance ids itself and throws when an add fails (menger#110); the
// opaque type menger kept to catch a -1 is now optix-jni's.
type InstanceId = io.github.lene.optix.InstanceId
