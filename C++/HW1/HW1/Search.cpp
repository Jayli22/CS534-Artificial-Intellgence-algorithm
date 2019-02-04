#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include <iomanip>
#include <map>
#include<algorithm>
using namespace std;

class Node
{
public:
	map<char, float> neighborNodes;
	//map<string, float> childNodes;
	float heuristicCost;
};

class  Search
{
public:
	map<char, Node> nodeMap;
	vector<string> queue;
	map<string, float> cost_queue;
	char curNode;
	void General_Search(string problem, string searchMethod);
	bool MakeNode(string problemname);

};

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

bool Search::MakeNode(string problemname)
{
	ifstream in(problemname);
	string filename;
	string line;
	if (in)
	{
		while (getline(in, line))
		{
			if (line[0] == '#')
			{
				break;
			}
			int i = 0;
			char pNode;
			char cNode;
			string sCost;
			pNode = line[0];



			i += 2;
			cNode = line[i];



			i += 2;
			while (i < line.length())
			{
				sCost += line[i];
				i++;
			}
			float cost = stringToNum<float>(sCost);
			if (nodeMap.find(pNode) != nodeMap.end())
			{
				Node p = nodeMap[pNode];
				Node c;
				if (nodeMap.find(cNode) != nodeMap.end())
				{
					c = nodeMap[cNode];
				}
				p.neighborNodes[cNode] = cost;
				c.neighborNodes[pNode] = cost;
				nodeMap[pNode] = p;
				nodeMap[cNode] = c;
			}
			else
			{
				Node p;
				Node c;
				if (nodeMap.find(cNode) != nodeMap.end())
				{
					c = nodeMap[cNode];
				}
				p.neighborNodes[cNode] = cost;
				c.neighborNodes[pNode] = cost;
				nodeMap[pNode] = p;
				nodeMap[cNode] = c;
			}
			cout << setiosflags(ios::fixed) << setiosflags(ios::right) << setprecision(1);




			//cout  << pNode << cNode << cost << endl;
			/*cout << pNode << "," << cNode << endl;
			cout << nodeMap[pNode].neighborNodes[cNode] << endl;
			cout << nodeMap['E'].neighborNodes['D'] << endl;*/

		}






		while (getline(in, line))
		{
			char node;
			string hcost;
			int i = 0;
			node = line[i];
			i += 2;
			while (i < line.length())
			{
				hcost += line[i];
				i++;
			}
			float cost = stringToNum<float>(hcost);
			nodeMap[node].heuristicCost = cost;
		}
		return true;
	}
	else // 没有该文件
	{
		cout << "no such file" << endl;
		return false;
	}
	//cout << nodeMap['B'].neighborNodes['A'] << endl;

}

bool Less_string(const string &str1, const string &str2)
{
	return str1.length() < str2.length();
}
bool More_string(const string &str1, const string &str2)
{
	return str1.length() > str2.length();
}

typedef pair<string, float> PAIR;
bool Less_cost(const PAIR& lhs, const PAIR& rhs) {
	return lhs.second < rhs.second;
}
bool Less_keystring(const PAIR& lhs, const PAIR& rhs) {
	return lhs.first.length() < rhs.first.length();
}
//struct CmpByValue {
//	bool operator()(const PAIR& lhs, const PAIR& rhs) {
//		return lhs.second < rhs.second;
//	}
//};
//
//struct CmpByKeyLength {
//	bool operator()(const string& k1, const string& k2) {
//		return k1.length() < k2.length();
//	}
//};

void Search::General_Search(string problem, string searchMethod)
{
	if (MakeNode(problem))
	{
		for (int searchnum = 1; searchnum <= 9; searchnum++)
		{
			switch (searchnum)
			{
			case 1:
				cout << " ·Depth 1st search" << "\n" << endl;

				break;
			case 2:
				cout << " ·Breadth 1st search" << "\n" << endl;

				break;

			case 3:
				cout << " ·Depth-limited search (depth-limit = 2)" << "\n" << endl;

				break;

			case 4:
				cout << " ·Iterative deepening search " << "\n" << endl;

				break;

			case 5:
				cout << " ·Uniform Search (Branch-and-bound) " << "\n" << endl;

				break;

			case 6:
				cout << " ·Greedy search " << "\n" << endl;

				break;

			case 7:
				cout << " ·A*" << "\n" << endl;

				break;

			case 8:
				cout << " ·Hill-Climbing" << "\n" << endl;

				break;

			case 9:
				cout << " ·Beam search (w = 2)" << "\n" << endl;

				break;

			}
			curNode = 'S';
			queue.push_back("S");
			nodeMap['S'].neighborNodes['S'] = 0;
			cost_queue["S"] = 0;
			cout << "   Expanded	" << "Queue" << endl;

			int search_complete = false;
			int search_failed = false;
			int L = 0;
			int W = 0;
			bool level_complete = false;
			bool eliminate_node = false;

			vector<PAIR> cost_queue_vector;
			cost_queue_vector.push_back(make_pair("S", nodeMap['S'].heuristicCost));
			while (nodeMap[curNode].neighborNodes.size() != 0)
			{
				if (search_complete)
				{
					break;
				}
				if (search_failed)
				{
					cout << "Search Failed!" << endl;
					break;
				}



				map<char, float>::iterator iter;
				iter = nodeMap[curNode].neighborNodes.begin();


				switch (searchnum)
				{
				case 1:
#pragma region Deep1st Search
				{
					if (curNode == 'S')
					{
						cout << "      " << curNode << "		" << "[<" << "S" << ">]" << endl;
					}
					string openQueue = queue[0];
					queue.erase(queue.begin());
					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;

							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}

					}

					// sort queue according different search method
					sort(queue.begin(), queue.end(), More_string);
					curNode = queue[0][0];
					cout << "      " << curNode << "		" << "[";
					for (int i = 0; i < queue.size(); i++)
					{
						cout << "<";
						cout << queue[i][0];

						for (int j = 1; j < queue[i].length(); j++)
						{
							cout << "," << queue[i][j];
						}
						cout << "> ";
					}
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}

				}

#pragma endregion

				break;
				case 2:
#pragma region Breadth1st Search
				{
					
					if (curNode == 'S')
					{
						cout << "      " << curNode << "		" << "[<" << "S" << ">]" << endl;
					}
					string openQueue = queue[0];
					queue.erase(queue.begin());
					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}

					}

					// sort queue according different search method
					sort(queue.begin(), queue.end(), Less_string);
					curNode = queue[0][0];
					cout << "      " << curNode << "		" << "[";
					for (int i = 0; i < queue.size(); i++)
					{
						cout << "<";
						cout << queue[i][0];

						for (int j = 1; j < queue[i].length(); j++)
						{
							cout << "," << queue[i][j];
						}
						cout << "> ";
					}
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}

					//}
				}
#pragma endregion
				break;
				case 3:
#pragma region Depth-limited search
				{

					string openQueue = queue[0];
					queue.erase(queue.begin());
					//string openQueue = queue[0];

					//queue.erase(queue.begin());

					//map<char, float>::iterator iter;
					//iter = nodeMap[curNode].neighborNodes.begin();
					if (curNode == 'S')
					{
						cout << "      " << curNode << "		" << "[<" << "S" << ">]" << endl;
					}

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							if (newQueue.length() > 3)
							{

								break;
							}
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}

					}

					if (queue.size() == 0)
					{
						search_failed = true;
						break;
					}
					// sort queue according different search method
					sort(queue.begin(), queue.end(), More_string);
					curNode = queue[0][0];
					cout << "      " << curNode << "		" << "[";
					for (int i = 0; i < queue.size(); i++)
					{
						cout << "<";
						cout << queue[i][0];

						for (int j = 1; j < queue[i].length(); j++)
						{
							cout << "," << queue[i][j];
						}
						cout << "> ";
					}
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;

						break;
					}

				}

#pragma endregion

				break;
				case 4:

#pragma region Interactive search
				{


					string openQueue = queue[0];
					queue.erase(queue.begin());
					if (curNode == 'S')
					{
						cout << "L=" << L << "   " << curNode << "		" << "[<" << "S" << ">]" << endl;
					}

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							if (newQueue.length() > L + 1)
							{
								if (queue.size() == 0)
								{
									L++;
									cout << "\n";
									break;
								}
								else
								{
									iter++;
								}

							}
							else
							{
								queue.push_back(newQueue);
								iter++;
							}

						}
						else
						{
							iter++;
						}

					}

					if (queue.size() == 0)
					{
						curNode = 'S';
						queue.push_back("S");
						break;
					}
					// sort queue according different search method
					sort(queue.begin(), queue.end(), More_string);
					curNode = queue[0][0];
					cout << "      " << curNode << "		" << "[";
					for (int i = 0; i < queue.size(); i++)
					{
						cout << "<";
						cout << queue[i][0];

						for (int j = 1; j < queue[i].length(); j++)
						{
							cout << "," << queue[i][j];
						}
						cout << "> ";
					}
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}

				}

#pragma endregion

				break;
				case 5:
#pragma region Uniform Search
				{
					vector<PAIR> cost_queue_vector(cost_queue.begin(), cost_queue.end());
					sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);

					string open_queue;
					float open_cost;
					curNode = cost_queue_vector[0].first[0];
					iter = nodeMap[curNode].neighborNodes.begin();
					cout << "      " << curNode << "		" << "[";



					for (int i = 0; i < cost_queue_vector.size(); i++)
					{
						cout << cost_queue_vector[i].second << "<";
						cout << cost_queue_vector[i].first[0];

						for (int j = 1; j < cost_queue_vector[i].first.length(); j++)
						{
							cout << "," << cost_queue_vector[i].first[j];
						}
						cout << "> ";
					}

					/*	for (int i = 0; i < cost_queue_vector.size(); i++)
						{
							cout << cost_queue_vector[i].second << "<" << cost_queue_vector[i].first << "> ";
						}*/
						/*map<string, float>::iterator titer = cost_queue.begin();
						while (titer != cost_queue.end())
						{
							cout <<titer->second<< "<" << titer->first << "> ";
							titer++;
						}*/
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}




					open_queue = cost_queue_vector[0].first;
					open_cost = cost_queue_vector[0].second;
					cost_queue.erase(open_queue);
					cost_queue_vector.erase(cost_queue_vector.begin());

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						/*if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}*/
						if (open_queue.find(childnode) == string::npos)
						{
							string newQueue = childnode + open_queue;
							cost_queue[newQueue] = open_cost + nodeMap[open_queue[0]].neighborNodes[childnode];
							cost_queue_vector.push_back(make_pair(newQueue, cost_queue[newQueue]));
							iter++;
						}
						else
						{
							iter++;
						}

					}

					// sort queue according different search method
					//sort(queue.begin(), queue.end(), );

					//sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);

					//curNode = queue[0][0];


					//}
				}
#pragma endregion

				break;
				case 6:
#pragma region Greedy Search
				{

					vector<PAIR> cost_queue_vector(cost_queue.begin(), cost_queue.end());
					sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);

					for (int i = 0; i < cost_queue_vector.size() - 1; i++)
					{
						if (cost_queue_vector[i].second == cost_queue_vector[i + 1].second)
						{
							if (cost_queue_vector[i].first.length() > cost_queue_vector[i + 1].first.length())
							{
								PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
								cost_queue_vector[i] = cost_queue_vector[i + 1];
								cost_queue_vector[i + 1] = t;
							}
						}
					}



					string open_queue;
					float open_cost;
					curNode = cost_queue_vector[0].first[0];
					if (curNode == 'S')
					{
						cost_queue_vector[0].second = nodeMap['S'].heuristicCost;
					}
					iter = nodeMap[curNode].neighborNodes.begin();
					cout << "      " << curNode << "		" << "[";

					for (int i = 0; i < cost_queue_vector.size(); i++)
					{
						cout << cost_queue_vector[i].second << "<";
						cout << cost_queue_vector[i].first[0];

						for (int j = 1; j < cost_queue_vector[i].first.length(); j++)
						{
							cout << "," << cost_queue_vector[i].first[j];
						}
						cout << "> ";
					}
					/*map<string, float>::iterator titer = cost_queue.begin();
					while (titer != cost_queue.end())
					{
						cout <<titer->second<< "<" << titer->first << "> ";
						titer++;
					}*/
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}




					open_queue = cost_queue_vector[0].first;
					open_cost = cost_queue_vector[0].second;
					cost_queue.erase(open_queue);
					cost_queue_vector.erase(cost_queue_vector.begin());

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						/*if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}*/
						if (open_queue.find(childnode) == string::npos)
						{
							string newQueue = childnode + open_queue;
							cost_queue[newQueue] = nodeMap[childnode].heuristicCost;
							cost_queue_vector.push_back(make_pair(newQueue, cost_queue[newQueue]));
							iter++;
						}
						else
						{
							iter++;
						}

					}

					// sort queue according different search method
					//sort(queue.begin(), queue.end(), );

					//sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);

					//curNode = queue[0][0];


					//}
				}
#pragma endregion
				break;
				case 7:
#pragma region A*
				{

					vector<PAIR> cost_queue_vector(cost_queue.begin(), cost_queue.end());
					for (int i = 0; i < cost_queue_vector.size() - 1; i++)
					{
						for (int j = i + 1; j < cost_queue_vector.size(); j++)
						{
							if (cost_queue_vector[i].second + nodeMap[cost_queue_vector[i].first[0]].heuristicCost >
								cost_queue_vector[j].second + nodeMap[cost_queue_vector[j].first[0]].heuristicCost)
							{

								PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
								cost_queue_vector[i] = cost_queue_vector[j];
								cost_queue_vector[j] = t;

							}
							if (cost_queue_vector[i].second == cost_queue_vector[j].second)
							{
								if (cost_queue_vector[i].first.length() > cost_queue_vector[j].first.length())
								{
									PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
									cost_queue_vector[i] = cost_queue_vector[j];
									cost_queue_vector[j] = t;
								}
							}
						}
					}


					string open_queue;
					float open_cost;
					curNode = cost_queue_vector[0].first[0];
					iter = nodeMap[curNode].neighborNodes.begin();
					cout << "      " << curNode << "		" << "[";



					for (int i = 0; i < cost_queue_vector.size(); i++)
					{
						cout << cost_queue_vector[i].second + nodeMap[cost_queue_vector[i].first[0]].heuristicCost << "<";
						cout << cost_queue_vector[i].first[0];

						for (int j = 1; j < cost_queue_vector[i].first.length(); j++)
						{
							cout << "," << cost_queue_vector[i].first[j];
						}
						cout << "> ";
					}


					/*	for (int i = 0; i < cost_queue_vector.size(); i++)
						{
							cout << cost_queue_vector[i].second + nodeMap[cost_queue_vector[i].first[0]].heuristicCost << "<" << cost_queue_vector[i].first << "> ";
						}*/
						/*map<string, float>::iterator titer = cost_queue.begin();
						while (titer != cost_queue.end())
						{
							cout <<titer->second<< "<" << titer->first << "> ";
							titer++;
						}*/
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}




					open_queue = cost_queue_vector[0].first;
					open_cost = cost_queue_vector[0].second;
					cost_queue.erase(open_queue);
					cost_queue_vector.erase(cost_queue_vector.begin());

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						bool isbest = true;
						char childnode = iter->first;
						/*if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}*/
						if (open_queue.find(childnode) == string::npos)
						{
							string newQueue = childnode + open_queue;
							cost_queue[newQueue] = open_cost + nodeMap[open_queue[0]].neighborNodes[childnode];

							map<string, float>::iterator titer = cost_queue.begin();
							while (titer != cost_queue.end())
							{
								if (cost_queue[newQueue] > titer->second && newQueue[0] == titer->first[0])
								{
									isbest = false;
									cost_queue.erase(newQueue);
									break;
								}
								if (cost_queue[newQueue] < titer->second && newQueue[0] == titer->first[0])
								{
									cost_queue.erase(titer);
									break;
								}
								titer++;
							}

							if (isbest)
							{
								cost_queue_vector.push_back(make_pair(newQueue, cost_queue[newQueue]));
								iter++;
							}
							else
							{
								iter++;
							}
						}
						else
						{
							iter++;
						}

					}

					// sort queue according different search method
					//sort(queue.begin(), queue.end(), );

					//sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);

					//curNode = queue[0][0];


					//}
				}
#pragma endregion
				break;
				case 8:
#pragma region Hill-Climb Search
				{
					if (curNode == 'S')
					{
						cost_queue["S"] = nodeMap['S'].heuristicCost;
					}
					vector<PAIR> cost_queue_vector(cost_queue.begin(), cost_queue.end());
					sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);
					if (cost_queue_vector.size() == 0)
					{
						search_failed = true;
						break;
					}
					for (int i = 0; i < cost_queue_vector.size() - 1; i++)
					{
						if (cost_queue_vector[i].second == cost_queue_vector[i + 1].second)
						{
							if (cost_queue_vector[i].first.length() > cost_queue_vector[i + 1].first.length())
							{
								PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
								cost_queue_vector[i] = cost_queue_vector[i + 1];
								cost_queue_vector[i + 1] = t;
							}
						}
					}



					
					string open_queue;
					float open_cost;
					curNode = cost_queue_vector[0].first[0];
					iter = nodeMap[curNode].neighborNodes.begin();
					cout << "      " << curNode << "		" << "[";


					for (int i = 0; i < cost_queue_vector.size(); i++)
					{
						if (cost_queue_vector[i].first.size() < cost_queue_vector[0].first.size())
						{
							cost_queue_vector.erase(cost_queue_vector.begin() + i);
							i--;
						}
						else
						{


							cout << cost_queue_vector[i].second << "<";
							cout << cost_queue_vector[i].first[0];

							for (int j = 1; j < cost_queue_vector[i].first.length(); j++)
							{
								cout << "," << cost_queue_vector[i].first[j];
							}

							cout << "> ";
						}
					}
					
					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}




					open_queue = cost_queue_vector[0].first;
					open_cost = cost_queue_vector[0].second;
					cost_queue.clear();
					cost_queue_vector.erase(cost_queue_vector.begin());

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						/*if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}*/
						if (open_queue.find(childnode) == string::npos)
						{
							string newQueue = childnode + open_queue;
							cost_queue[newQueue] = nodeMap[childnode].heuristicCost;
							cost_queue_vector.push_back(make_pair(newQueue, cost_queue[newQueue]));
							iter++;
						}
						else
						{
							iter++;
						}

					}



				}
#pragma endregion
				break;
				case 9:
#pragma region Beam Search
				{

					//cost_queue_vector(cost_queue.begin(), cost_queue.end());
					if (cost_queue_vector.size() == 0)
					{
						break; 
						search_failed = true;
					}
					sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_keystring);
					if (level_complete && cost_queue_vector.size() > 2)
					{
						sort(cost_queue_vector.begin(), cost_queue_vector.end(), Less_cost);



						for (int i = 2; i < cost_queue_vector.size(); )
						{
							cost_queue_vector.erase(cost_queue_vector.begin() + i);
						}

						for (int i = 0; i < cost_queue_vector.size(); i++)
						{
							for (int j = i + 1; j < cost_queue_vector.size(); j++)
							{
								for (int p = 0; p < cost_queue_vector[i].first.size(); p++)
								{
									if (cost_queue_vector[i].first[p] > cost_queue_vector[j].first[p])
									{
										PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
										cost_queue_vector[i] = cost_queue_vector[j];
										cost_queue_vector[j] = t;
									}
								}
							}
						}
						level_complete = false;
					}
					else
					{
						level_complete = false;

					}
					for (int i = 0; i < cost_queue_vector.size() - 1; i++)
					{
						if (cost_queue_vector[i].second == cost_queue_vector[i + 1].second)
						{
							/*if (cost_queue_vector[i].first.length() > cost_queue_vector[i + 1].first.length())
							{
								PAIR t = make_pair(cost_queue_vector[i].first, cost_queue_vector[i].second);
								cost_queue_vector[i] = cost_queue_vector[i + 1];
								cost_queue_vector[i + 1] = t;
							}*/
						}
					}



					string open_queue;
					float open_cost;
					curNode = cost_queue_vector[0].first[0];
					iter = nodeMap[curNode].neighborNodes.begin();
					cout << "      " << curNode << "		" << "[";


					if (cost_queue_vector.size() < 2)
					{
						level_complete = true;
					}
					else if (cost_queue_vector[0].first.length() < cost_queue_vector[1].first.length())
					{
						level_complete = true;
					}



					for (int i = 0; i < cost_queue_vector.size(); i++)
					{
						cout << cost_queue_vector[i].second << "<";
						cout << cost_queue_vector[i].first[0];

						for (int j = 1; j < cost_queue_vector[i].first.length(); j++)
						{
							cout << "," << cost_queue_vector[i].first[j];
						}
						cout << "> ";
					}




					cout << "]" << endl;
					if (curNode == 'G')
					{
						cout << "goal reached!" << endl;
						search_complete = true;
						break;
					}




					open_queue = cost_queue_vector[0].first;
					open_cost = cost_queue_vector[0].second;
					cost_queue_vector.erase(cost_queue_vector.begin());

					while (iter != nodeMap[curNode].neighborNodes.end())
					{
						char childnode = iter->first;
						/*if (openQueue.find(childnode) == string::npos)
						{
							string newQueue = childnode + openQueue;
							queue.push_back(newQueue);
							iter++;
						}
						else
						{
							iter++;
						}*/
						if (open_queue.find(childnode) == string::npos)
						{
							string newQueue = childnode + open_queue;
							//cost_queue[newQueue] = nodeMap[childnode].heuristicCost;
							cost_queue_vector.push_back(make_pair(newQueue, nodeMap[childnode].heuristicCost));
							iter++;
						}
						else
						{
							iter++;
						}

					}



				}
#pragma endregion
				break;
				default:
					break;
				}


			}
			queue.clear();
			cost_queue.clear();
			cost_queue_vector.clear();
		}
	}

}



int main()
{
	Search s;
	string input;
	cout << "Please input input file name:" << endl;
	cin >> input;

	s.General_Search(input, "S");
	//s.General_Search("graph2.txt", "S");

	system("pause");
	return 0;

}
