### Build up the Framework 
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. It will include the following features:
1. Text/Image Conditioned Mesh Generation. Inputs: Text or a single image or multiple images, and a mesh; Output: a textured mesh
2. Text/Image Conditioned Texture Generation.  Inputs: Text or a single image or multiple images, and a mesh; Output: a textured mesh
3. Mesh Segmentation,  Inputs: a GLB mesh, Outputs: a GLB mesh (scene)
4.Auto-Rig the mesh . Inputs: A OBJ/GLB/FBX mesh without rig/armature, Outputs:  A OBJ/GLB/FBX mesh with rig/armature
There can be multiple models supporting the same feature, and each model will be marked with a required VRAM value. The involved input formats include text/image(s)/meshes. Possible formats for mesh models are OBJ/FBX/GLB. Now I hope you help me design a FastAPI server framework/system that includes:
1. The abstraction layers on the model/feature implementation
2. Main server entry that directly interacts with the user (receiving and addressing their requests)
3. Backend scheduler for the AI models (loading and offload when required VRAM is NOT enough). 
DON'T hurry to implement any details, focus on the system design. Make sure it's scalable and extensible.  Follow the best practice in AI inference and system design. Output a draw.io XML framework, and also detailed documents under #docs

### Fix the bug of concurreny
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Currently I am encountering a bug in terms of the model scheduler and the model processing concurrency. Specifically, when multiple incoming requests request the SAME model, it may exceed the model capacibity. The bug is possibly because of that in finding the models for the request #_find_best_model you didn't consider the model capacity, or you may add some locks somewhere to avoid the race condition. The bug is a bit handy and you should be very careful to find out the real reason and fix it.

Implement model specificaitions
Good work! Now the problem is fixed. Please move on to implement the model specifications of all mesh generation methods, mesh segmentation models, auto rig models according to the implementation guide #file:implementation_guide.md . You should implement every thing except the model adpaters (which I will implement later). Be careful and ensure that your code is well-organized and following the best practice of engineering. Extra info:
1. Text/Image Conditioned Mesh Generation. Inputs: Text or a single image or multiple images, and a mesh; Output: a textured mesh
2.Text/Image Conditioned Texture Generation.  Inputs: Text or a single image or multiple images, and a mesh; Output: a textured mesh
3.Mesh Segmentation,  Inputs: a GLB mesh, Outputs: a GLB mesh (scene)
4.Auto-Rig the mesh . Inputs: A OBJ/GLB/FBX mesh without rig/armature, Outputs:  A OBJ/GLB/FBX mesh with rig/armature
Follow my instructions but don't make things over-complicated. The rig models and segment models should not have subclaseses. The mesh models and texture models can have different subclasses that correspond to different condition types.

### Question 
According to the codebase, I have some questions. If I have multiple models running on different GPUs, can them be executed in parallel? Throughout the codebase, I can only see coroutines, but not spawned processed or threads. Besides, will there be any problem for the multi-processing of Pytorch Spawning?

### Fix TRELLIS test script
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase. But one of the testing logic is incorrect #file:test_trellis_adapter.py . Please rewrite that test script. You should NOT use ANY mock and should DO true GPU inference, no need to test the vram requirement, status, feature type, simply input images/text and ensure the pipeline can run. Example assets are provided at #assets. DON'T MODIFY ANY IRRELEVANT FILE.

### Implement real tests
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md  . Also a number of test scripts for the adpaters are implemented under @/test_adapters , but all of them directly tests the adapter, igoring the pipeline and API service. Now I need you to implement some real API tests under @/test_integration  for each adapter, with reference to the trellis test @test_trellis_adapter_integration.py  . You should check out the routers @routers as well as the adapter implmenetations.

### Fix Adapter Registry
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md  . A number of adapter tests or integration tests are provided under @/tests . But I have found something unreasonable in current design. Specifically, in the @lifespan of main entrypoint, it gets the adapter registry which creates all adapters. The adapter instances to create are hard-coded around @adapter_registry.py  , which means the settings and configs @models.yaml not working or not used and request to query the available features simply get empty list. Please improve current design on model/adapter registration. But DON'T MODIFY TOO MUCH IRRELEVANT CODE.

### Multiprocess Pytorch Inference
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md   . A number of adapter tests or integration tests are provided under @/tests . However, current codebase (@/adapters and @model_scheduler.py  ) extensively use asyncio and coroutine, which means no TRUE parallelism is there. As I many have multiple GPUs or run multiple workers of the same model, multiprocessing inference is critical in production. You can refer to @pytorch_multiprocess_inference.md   for some information about the pytorch multiprocessing inference design, carefuly addressing the model scheduler, worker spawn and configuration. DON'T MODIFY IRRELEVANT CODE.

### Discard Model Async Concurrency Parameter
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md   . Now current codebase supports two types of schedulers at @scheduler_factory.py. I need you to completely remove the asyncio scheduler and exclusively use the multiprocessing scheduler  . Also I need you to discard the logics about concurrency/capacity of all the @adapters and @models , as the AI workers will always execute in the blocking mode when processing an incoming request. Be careful and don't modify irrelevant code.

### Model Preference 
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md . Now I have noticed that the model preference specified in job request @JobRequest are used nowhere. In my expectation, it should be verified in the multiprocessing scheduler @multiprocess_scheduler.py . Especially you should look up this value in @_find_worker_for_job , where `model_preference` is simply `model_id` there. Besides, in each router function under @routers, you should also verify the specified model_preference is in the available registered models of the scheduler. DON'T MODIFY IRRELEVANT CODE.

### Update Multiprocessing Scheduler Logics 
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md . Currently the multiprocessing scheduler @multiprocess_scheduler.py has some problems that I need you to fix. Specifically, the scheduler should NOT create all workers at the beginning, instead it should create workers on demand. Here are more details:
1.Relationships between model_id and worker: each model_id can have multiple workers, and in the system configuration we will specify the maximum workers for each model_id (can be different for each model). 
2.When a request is coming asking for some model_id, you should create a worker for it ONLY when:
(1)All existing workers serving for that model_id are busy, or NO WORKER for that model_id exists currently.
(2)There are sufficient GPU memory on any GPU (required by `vram` parameter in each model_id configuration)
(3)The number of workers for that model_id IS NOT up to maximum value.
If the above conditions are all satisifed, you should launch a worker for that model_id and load the model(pretrained weights) on that worker . Otherwise, you should keep the request in queue and wait until any of the above conditions is satisfiled. 
3.From the perspective of the worker management, if any worker has finished processing and becomes idle, and there are SOME request in the queue waiting for VRAM, the worker should be destroyed to release VRAM.
To accomplish these goals, you may need to refer to some of the GPU monitor @gpu_monitor.py if necessary 

### Improve and fix GPU Monitor and Scheduler 
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md . Currently the multiprocessing scheduler @multiprocess_scheduler.py and GPU monitor @gpu_monitor.py   has some logics that I need you to update. Specifically, each model will claim about how much VRAM it requires, and the scheduler will create new worker for that model ONLY when there is sufficient GPU VRAM. But currently the scheduler querys the total idle GPU VRAM across all GPUs, which is INCORRECT . You can assume that each model will ONLY occupy a single ONE GPU. Besides, currently the GPU monitor finds idle vram in real time by checking idle GPU VRAM, I need you to implement another monitor mode, that simply deducts the `vram_requirement` requested by a model from the total VRAM of a specific GPU and increment it once a worker is cleaned up. Please update the files according to my requirements. DON'T MODIFY IRRELEVANT CODE.

### Write README 
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase. I need you to read through the codebase, write a README for this project. The README should be concise but outline main system architecture, features and supported models of this project.

### Write API DOCS
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase. I need you to read through the codebase (especially the @routers), and provide comprehensive interface docs (API docs) of current system, including the url of each supported endpoint, its functionality, its required input parameters, and the expected outputs. Provided the summarized API documents under @docs.

### Configure Logging yaml
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models.  Now the framework is almost done in the current codebase. But I have noticed that the logging system is NOT configured correctly. The configuration @logging.yaml is NOT used anywhere, and in @main.py, or other places, it DOES not setup any logging to any file. Please help me fix.

### Enhance test client
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase. Now there is a comprehensive test client @run_test_client.py   which tests the server via web request (suppose it's already started). But current test client always waits until a job is finished before it submits a new job. Now I need you to strengthen the test client to support submit all the jobs at the beginning, and then keep monitering all the job status until all of them are done or failed. DON'T MODIFY IRRELETANT CODE.

### Scheduler Logic
You are an expert at Python, AI inference and system design. Now I am developing a backend server for 3D generative models. Now the framework is almost done in the current codebase according to the implementation guide @implementation_guide.md . Currently the multiprocessing scheduler @multiprocess_scheduler.py has some problems that I need you to update. Specifically, when an incoming request(dequeued from the job queue) can NOT be processed (e.g. because all workers are BUSY or there is NOT sufficient VRAM), it plans serveral retry in the future (with incrementing delays), and when all the trials failed to get processed, it fails the job. This logic is NOT expected, actually you should keep it always in the queue until it's processed. And before it gets processed, the job submitted after it will NEVER get processed. Unless there is NO chance that this job get processed(e.g. required VRAM more than the total GPU VRAM, or 1 hour has passed). Fix this logic.
