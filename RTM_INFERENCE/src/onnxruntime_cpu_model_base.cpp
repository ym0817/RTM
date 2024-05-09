#include "onnxruntime_cpu_model_base.h"

// #include "characterset_convert.h"

OnnxruntimeCPUModelBase::OnnxruntimeCPUModelBase()
	:m_env(nullptr),
	m_session(nullptr)
{
}

OnnxruntimeCPUModelBase::~OnnxruntimeCPUModelBase()
{
}

bool OnnxruntimeCPUModelBase::LoadModel(const std::string& onnx_env_name, const std::string& onnx_model_path)
{
	// std::wstring onnx_model_path_wstr = stubbornhuang::CharactersetConvert::string_to_wstring(onnx_model_path);

	m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, onnx_env_name.c_str());

	if (m_env == nullptr)
		return false;

	int cpu_processor_num = std::thread::hardware_concurrency();
	cpu_processor_num /= 2;

	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(cpu_processor_num);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetLogSeverityLevel(4);

	OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
	m_session = Ort::Session(m_env, onnx_model_path.c_str(), session_options);

	if (m_session == nullptr)
		return false;

	return true;
}

void OnnxruntimeCPUModelBase::PrintModelInfo(Ort::Session& session)
{
	// print the number of model input nodes
//���ģ������ڵ������
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();
	std::cout << "Number of input node is:" << num_input_nodes << std::endl;
	std::cout << "Number of output node is:" << num_output_nodes << std::endl;

	// print node name
	//��������Ľڵ���
	Ort::AllocatorWithDefaultOptions allocator;
	std::cout << std::endl;//�������
	for (auto i = 0; i < num_input_nodes; i++)
		// std::cout << "The input op-name " << i << " is:" << session.GetInputNameAllocated(i, allocator) << std::endl;
	for (auto i = 0; i < num_output_nodes; i++)
		// std::cout << "The output op-name " << i << " is:" << session.GetOutputNameAllocated(i, allocator) << std::endl;

	// ��ȡ�����������

	// print input and output dims
	//��ȡ�������ά��
	for (auto i = 0; i < num_input_nodes; i++)
	{
		std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "input " << i << " dim is: ";
		for (auto j = 0; j < input_dims.size(); j++)
			std::cout << input_dims[j] << " ";
	}
	for (auto i = 0; i < num_output_nodes; i++)
	{
		std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "output " << i << " dim is: ";
		for (auto j = 0; j < output_dims.size(); j++)
			std::cout << output_dims[j] << " ";
	}
}
