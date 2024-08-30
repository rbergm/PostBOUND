SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE v.UserId = u.Id
  AND c.UserId = u.Id
  AND ph.UserId = u.Id
  AND c.CreationDate <= CAST('2014-08-28 07:25:55' AS timestamp)
  AND ph.PostHistoryTypeId = 2
  AND u.Reputation >= 1
  AND u.Views >= 0
  AND u.DownVotes >= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 15
  AND u.CreationDate >= CAST('2010-09-03 11:45:16' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-18 17:19:53' AS timestamp);
