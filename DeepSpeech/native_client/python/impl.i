%module(threads="1") impl

%{
#define SWIG_FILE_WITH_INIT
#include "deepspeech.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

// apply NumPy conversion typemap to DS_FeedAudioContent and DS_SpeechToText
%apply (short* IN_ARRAY1, int DIM1) {(const short* aBuffer, unsigned int aBufferSize)};

// apply NumPy conversion typemap to DS_AudioToInputVector
%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(float** aMfcc, int* aNFrames, int* aFrameLen)};

%typemap(in, numinputs=0) ModelState **retval (ModelState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) ModelState **retval {
  // not owned, Python wrapper in __init__.py calls DS_DestroyModel
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(in, numinputs=0) StreamingState **retval (StreamingState *ret) {
  ret = NULL;
  $1 = &ret;
}

%typemap(argout) StreamingState **retval {
  // not owned, DS_FinishStream deallocates StreamingState
  %append_output(SWIG_NewPointerObj(%as_voidptr(*$1), $*1_descriptor, 0));
}

%typemap(newfree) char* "free($1);";
%newobject DS_SpeechToText;
%newobject DS_IntermediateDecode;
%newobject DS_FinishStream;

%rename ("%(strip:[DS_])s") "";

%include "../deepspeech.h"
