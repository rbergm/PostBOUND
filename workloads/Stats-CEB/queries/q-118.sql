SELECT COUNT(*)
FROM postHistory AS ph,
  posts AS p,
  votes AS v,
  users AS u
WHERE u.Id = p.OwnerUserId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND ph.CreationDate <= CAST('2014-07-28 13:25:35' AS timestamp)
  AND p.PostTypeId = 1
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 4
  AND v.CreationDate >= CAST('2010-07-20 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-03 00:00:00' AS timestamp)
  AND u.DownVotes = 0
  AND u.CreationDate <= CAST('2014-08-08 07:03:29' AS timestamp);
